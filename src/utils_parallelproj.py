'''
This scripts contains the functions to reconstruct the PET frames and fit 
noisy washout rate maps from the PET listmode data
'''

import struct
import gc
import array_api_compat.cupy as xp
import parallelproj
from array_api_compat import to_device
import array_api_compat.numpy as np
from cupyx.scipy.ndimage import median_filter, convolve
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm


def interpolate_nan_values(arr, max_iterations=10):
    ''' For voxels with insufficient PET counts to reliably fit the decay curve 
    values are computed by averaging neighboring voxels within a 3x3x3 kernel.
    '''
    # Create a mask of NaN values
    nan_mask = xp.isnan(arr)

    # Replace NaNs with zero for initial computation
    arr_filled = arr.copy()
    arr_filled[nan_mask] = 0

    # Create a weights array where non-NaNs are 1 and NaNs are 0
    weights = xp.ones_like(arr)
    weights[nan_mask] = 0

    # Define a convolution kernel (3x3x3 cube)
    kernel = xp.ones((3, 3, 3))

    for _ in range(max_iterations):
        # Convolve the filled array and weights with the kernel
        sum_values = convolve(arr_filled, kernel, mode="constant", cval=0.0)
        sum_weights = convolve(weights, kernel, mode="constant", cval=0.0)

        # Avoid division by zero
        arr_interp = xp.zeros_like(arr_filled)
        non_zero_weights = sum_weights != 0
        arr_interp[non_zero_weights] = (
            sum_values[non_zero_weights] / sum_weights[non_zero_weights]
        )

        # Update only the positions that are still NaN
        new_nan_mask = xp.isnan(arr_filled)
        arr_filled[new_nan_mask] = arr_interp[new_nan_mask]
        weights[new_nan_mask] = 1  # Mark these positions as filled

        # Break if no NaNs are left
        if not xp.isnan(arr_filled).any():
            break

    return arr_filled


def decay_function(x, a, b):
    ''' Exponential decay function for curve fitting. '''
    return a * np.exp(-b * x)


def lm_em_update(x_cur, op, adjoint_ones):
    """Update the image estimate using the EM algorithm"""
    epsilon = 1e-10  #  If ybar contains zeros, dividing by it can produce NaNs
    ybar = op(x_cur)
    x = x_cur * op.adjoint(1 / (ybar + epsilon)) / adjoint_ones
    return x


def dynamic_decay_reconstruction(
    psf_path,
    img_shape=(248, 140, 176),
    voxel_size=(1.9531, 1.9531, 1.5),
    scanner="vision",
    num_subsets=2,
    osem_iterations=3,
    body_mask=None,  # array with the same shape as the image, 1 for body, 0 for air
    sensitivity_array=None,
    frame_duration=3,  # in minutes
    end_time=30,  # in minutes
    reduction_factor=1,
    tumor_xmin=0,
    tumor_xmax=-1,
    tumor_ymin=0,
    tumor_ymax=-1,
    tumor_zmin=0,
    tumor_zmax=-1,
    detector_center_z=0.0,  # center of the scanner in z
    tumor_mask_dilated=None,
    minimum_mean_life=1e-5,
    maximum_mean_life=1e5,
    patient_info_path=None,
    fixed_decay_rate=None,
    save_gif_frames=False,
):
    """
    Reconstruction from listmode using parallelproj library. Adding realistic effects to the PET data.
    The washout rate and initial activity are fitted to the reconstructed PET frames.
    """
    # choose a device (CPU or CUDA GPU)
    if "numpy" in xp.__name__:
        # using numpy, device must be cpu
        dev = "cpu"
    elif "cupy" in xp.__name__:
        # using cupy, only cuda devices are possible
        dev = xp.cuda.Device(0)
    elif "torch" in xp.__name__:
        # using torch valid choices are 'cpu' or 'cuda'
        dev = "cuda"

    if scanner == "quadra":
        # Create a Quadra PET scanner
        num_rings = 320
        radius = 410.0  # mm
        max_z = 530.0
        num_sides = 760
        # TOF
        TOF_resolution = 225
        psf = 3.5  # resolution of the scanner in mm
        depth = 20  # mm depth of crystal (depth is the max DOI if all photons are incident perpendicular to the crystal)
        mu = 0.082  # mm^-1 attenuation coefficient
        FWHM_detector = (
            3.39  # mm (given by detector side length, 2*pi*radius/num_sides)
        )
    elif scanner == "vision":
        # Create a Vision PET scanner
        radius = 410.0  # mm
        num_rings = 80
        max_z = 131.15
        num_sides = 760
        # TOF
        TOF_resolution = 225
        psf = 3.5  # resolution of the scanner in mm
        depth = 20  # mm depth of crystal (depth is the max DOI if all photons are incident perpendicular to the crystal)
        mu = 0.082  # mm^-1 attenuation coefficient
        FWHM_detector = (
            3.39  # mm (given by detector side length, 2*pi*radius/num_sides)
        )
    else:
        raise ValueError("Invalid scanner type")

    tofbin_FWHM = (
        TOF_resolution * 1e-12 * 3e8 / 2 * 1e3
    )  # *1e3 to mm;  *1e-12 to s; *3e8 to m/s;  /2 to get one-way distance;
    sigma_tof = tofbin_FWHM / 2.355  # to get sigma from FWHM
    tofbin_width = (
        sigma_tof * 1.03
    )  # sigma_tof * 1.03, as given in https://parallelproj.readthedocs.io/en/stable/python_api.html#module-parallelproj.tof # ps, it is the minimum time difference between the arrival of two photons that it can detect. it is diveded by 2 because if one of them arrivs TOF_resolution

    num_tofbins = 201
    if num_tofbins % 2 == 0:
        num_tofbins -= 1
    print("num_tofbins", num_tofbins)
    tof_params = parallelproj.TOFParameters(
        sigma_tof=sigma_tof, num_tofbins=num_tofbins, tofbin_width=tofbin_width
    )
    enable_tof = True

    # Blurring due to detector resolution, crystal size, DOI, positron range
    res_model = parallelproj.GaussianFilterOperator(
        img_shape, sigma=psf / (2.35 * xp.asarray(voxel_size))
    )

    adjoint_ones = to_device(xp.asarray(sensitivity_array, dtype=xp.float32), dev)

    # Define the structure format for one data record
    format_string = "Q f i f f f f f f h h"
    record_size = struct.calcsize(format_string)

    # Define the dtype for numpy based on the format string
    dtype = np.dtype(
        [
            ("emission_time", "u8"),  # unsigned long long int (Q) (emission_time (ps))
            ("travel_time", "f4"),  # float (f) (travel_time (ps))
            ("emission_voxel", "i4"),  # int (i) (emission voxel)
            ("energy", "f4"),  # float (f) (energy)
            ("z", "f4"),  # float (f) (z (cm))
            ("phi", "f4"),  # float (f) (phi (rad))
            (
                "vx",
                "f4",
            ),  # float (f) (vx; x component of the incident photon direction)
            (
                "vy",
                "f4",
            ),  # float (f) (vy; y component of the incident photon direction)
            (
                "vz",
                "f4",
            ),  # float (f) (vz; z component of the incident photon direction)
            (
                "index1",
                "i2",
            ),  # short int (h)  Flag for scatter: =0 for non-scattered, =1 for Compton, =2 for Rayleigh, and =3 for multiple scatter)
            ("index2", "i2"),  # short int (h) (index2)
        ]
    )

    # Read the binary file in chunks and convert directly to DataFrame
    chunk_size = 100000
    events = []
    with open(psf_path, "rb") as file:
        while True:
            data = file.read(record_size * chunk_size)
            if not data:
                break
            chunk = np.frombuffer(data, dtype=dtype)
            event = pd.DataFrame(chunk).loc[
                :, ["emission_time", "travel_time", "z", "phi", "vx", "vy", "vz"]
            ]
            events.append(event)

    del chunk, event, data
    events = pd.concat(events, ignore_index=True)

    num_coincidences = events.shape[0] // 2
    if patient_info_path is not None:
        with open(patient_info_path, "a") as patient_info_file:
            patient_info_file.write(
                f"\nFor scanner {scanner}, number of coincidences: {num_coincidences}\n"
            )

    # Dynamic PET: let's split the events into 10 X-minute frames
    start_time = 0
    end_time = end_time * 60 * 1e12  # in picoseconds
    frame_duration_ps = frame_duration * 60 * 1e12  # in picoseconds
    frame_edges = np.arange(start_time, end_time + frame_duration_ps, frame_duration_ps)
    frame_labels = np.arange(len(frame_edges) - 1)
    frame_times = (
        (frame_edges[:-1] + frame_edges[1:]) / 2 / 1e12
    )  # in seconds, taking midpoints of the frames
    events["frame"] = pd.cut(
        events["emission_time"], bins=frame_edges, labels=frame_labels, right=False
    )
    num_frames = len(frame_labels)
    if patient_info_path is not None:
        with open(patient_info_path, "a") as patient_info_file:
            patient_info_file.write(f"Number of events: {events.shape[0]}\n")
    if (
        tumor_mask_dilated is not None
    ):  # since there is still information surrounding the tumor, so know everything outside a dilation will be 0
        tumor_mask_dilated = tumor_mask_dilated[
            tumor_xmin:tumor_xmax, tumor_ymin:tumor_ymax, tumor_zmin:tumor_zmax
        ]
    else:
        raise ValueError("tumor_mask_dilated is required")
    reconstructed_images = []
    all_events = events.copy()

    if save_gif_frames:
        frame_step = 15  # 6 minute frames computed every 15 seconds
        num_frames = (60 / frame_step) * (
            end_time / 60 / 1e12 - frame_duration - start_time
        )  # first one from 0'0'' to 6'00'', next from 0'15'' to 6'15'', until 24 to 30
        num_frames = int(num_frames)

    for num_frame in range(num_frames):
        if save_gif_frames:
            print("Frame:", num_frame)
            events = all_events[
                (all_events["emission_time"] >= num_frame * frame_step * 1e12)
                & (
                    all_events["emission_time"]
                    < (num_frame * frame_step + frame_duration * 60) * 1e12
                )
            ]
        else:
            events = all_events[all_events["frame"] == num_frame]
        num_events = events.shape[0]
        travel_time = xp.asarray(events.travel_time)
        vx = xp.asarray(events.vx)
        vy = xp.asarray(events.vy)
        vz = xp.asarray(events.vz)
        phi = xp.asarray(events.phi)
        events_x = radius * xp.cos(phi)
        events_y = radius * xp.sin(phi)
        events_z = xp.asarray(events.z)

        # 1. DOI effect
        # Accounting for the angle of incidence of the photon, larger DOI if incidence angle is larger
        # We are getting the length of the path inside the detectors of each photon, solving:
        # r_salida (posicion donde sale del scanner) = r_vec (posicion donde llega al scanner) + v (direccion incidente) * DOI (escalar); sqrt(r_salida(0)**2 + r_salida(1)**2) = R+20mm (radio del escaner + anchura del cristal, ya que sale en el borde del cristal)

        max_dois = (
            1
            / (vx**2 + vy**2)
            * (
                -(vx * events_x + vy * events_y)
                + xp.sqrt(
                    (vx * events_x + vy * events_y) ** 2
                    - (vx**2 + vy**2)
                    * (events_x**2 + events_y**2 - (radius + depth) ** 2)
                )
            )
        )
        del events

        uniform_rands = xp.random.uniform(0, 1, num_events)  # one sample per event

        # Cumulative distribution function capturing that the probability of
        # a photon being detected is higher when photons enter the crystal
        # and decreases exponentially as they move into the crystal, considering a maximum depth of d and normalizing
        # F(x) = 1 - exp(-mu*x) / (1 - exp(-mu*max_doi))
        # Then, inverse transform sampling to sample values from the distribution
        # Sampling from this inverse function means picking random probabilities and finding the points on the distribution that correspond to those probabilities
        # (inverse of the CDF: F^-1(u) = -ln(1 - u(1 - exp(-mu*d))) / mu;

        dois = -np.log(1 - uniform_rands * (1 - xp.exp(-mu * max_dois))) / mu
        del uniform_rands

        # 2. Detector resolution effect:
        # The detector resolution is modeled as a Gaussian distribution with a FWHM of 3.39 mm
        # Values are sampled from a normal distribution with FWHM = 3.39 mm and added to the x, y, and z coordinates of the events
        sigma_detector = FWHM_detector / 2.355
        mean = 0.0
        event_displacement = xp.random.normal(mean, sigma_detector, num_events * 3)

        events_x = (
            events_x + event_displacement[:num_events] + vx * dois
        )  # x position of the detector
        events_y = (
            events_y + event_displacement[num_events : 2 * num_events] + vy * dois
        )  # y position of the detector
        events_z = (
            events_z * 10.0
            + event_displacement[2 * num_events : 3 * num_events]
            + vz * dois
        )  # *10.0 for cm to mm

        # 3. Crystal size effects:
        # the angle and z position of the event are rounded to the position of the crystal
        crystal_phi_positions = (
            xp.linspace(0, 2 * xp.pi, num_sides, endpoint=False) - np.pi
        )
        events_phi = xp.arctan2(
            events_y, events_x
        )  # angle of the event once DOI and detector resolution are considered
        events_phi[events_phi > xp.pi - 2 * xp.pi / num_sides] -= (
            2 * xp.pi
        )  # wrap around, so that if it wont be assigned to the bin below when it should be assigned to phi=-pi
        events_phi = (
            xp.digitize(events_phi, crystal_phi_positions) * 2 * xp.pi / num_sides
            - xp.pi
        )  # round to the closest crystal phi position
        events_x = radius * xp.cos(
            events_phi
        )  # x position of the crystal corresponding to the event
        events_y = radius * xp.sin(
            events_phi
        )  # y position of the crystal corresponding to the event
        crystal_z_positions = xp.linspace(-max_z, max_z, num_rings)
        events_z = (
            xp.digitize(xp.asarray(events_z), crystal_z_positions)
            * 2
            * max_z
            / num_rings
            + detector_center_z
            - max_z
        )  # round to the closest crystal z position

        # TOF bin
        bin = xp.round(
            (travel_time[0::2] - travel_time[1::2]) / (TOF_resolution / 2.355 * 1.03)
        ).astype(
            int
        )  # / 2.355 * 1.03 to match the spatial tof_bin width
        bin = xp.repeat(bin, 2)
        del (
            crystal_phi_positions,
            crystal_z_positions,
            events_phi,
            event_displacement,
            dois,
            travel_time,
            vx,
            vy,
            vz,
            max_dois,
            phi,
        )

        event_start_coordinates = xp.asarray(
            xp.stack((events_x[0::2], events_y[0::2], events_z[0::2]), axis=1)
        )
        event_end_coordinates = xp.asarray(
            xp.stack((events_x[1::2], events_y[1::2], events_z[1::2]), axis=1)
        )
        event_tof_bins = bin[0::2]
        del events_x, events_y, events_z, bin

        lm_proj = parallelproj.ListmodePETProjector(
            event_start_coordinates, event_end_coordinates, img_shape, voxel_size
        )

        if enable_tof:
            lm_proj.tof_parameters = tof_params
            lm_proj.event_tofbins = event_tof_bins
            lm_proj.tof = enable_tof

        subset_slices = [slice(i, None, num_subsets) for i in range(num_subsets)]

        lm_pet_subset_linop_seq = []

        for i, sl in enumerate(subset_slices):
            subset_lm_proj = parallelproj.ListmodePETProjector(
                event_start_coordinates[sl, :],
                event_end_coordinates[sl, :],
                img_shape,
                voxel_size,
            )

            # enable TOF in the LM projector
            subset_lm_proj.tof_parameters = lm_proj.tof_parameters
            if lm_proj.tof:
                subset_lm_proj.event_tofbins = 1 * event_tof_bins[sl]
                subset_lm_proj.tof = lm_proj.tof

            lm_pet_subset_linop_seq.append(
                parallelproj.CompositeLinearOperator((subset_lm_proj, res_model))
            )

        del event_start_coordinates, event_end_coordinates, event_tof_bins

        lm_pet_subset_linop_seq = parallelproj.LinearOperatorSequence(
            lm_pet_subset_linop_seq
        )

        # number of OSEM iterations
        num_iter = osem_iterations
        beta = 0.0  # 0.05
        # adjoint_ones = adjoint_ones + adjoint_ones.max()
        x = xp.ones(img_shape, dtype=xp.float32, device=dev)

        for i in range(num_iter):
            for k, sl in enumerate(subset_slices):
                print(
                    f"OSEM iteration {(k+1):03} / {(i + 1):03} / {num_iter:03}",
                    end="\r",
                )
                x = lm_em_update(
                    x,
                    lm_pet_subset_linop_seq[k],
                    adjoint_ones / num_subsets,
                )
                x = (1.0 - beta) * x + beta * median_filter(x, size=3)

        # Using the sensitivity as a calibration factor to obtain the activity
        # first, we normalize the image to the number of counts per second, then we divide by the sensitivity (calibrated) to get the activity
        num_counts = num_events // 2
        x = x / x.sum() * (num_counts / (frame_duration * 60)) / adjoint_ones

        if body_mask is not None:
            x[~body_mask] = 0
        x = x[tumor_xmin:tumor_xmax, tumor_ymin:tumor_ymax, tumor_zmin:tumor_zmax]
        reconstructed_images.append(x)
        for subset in lm_pet_subset_linop_seq:
            del subset
        del lm_pet_subset_linop_seq, lm_proj, subset_lm_proj

    if save_gif_frames:
        for i, reconstructed_image in enumerate(reconstructed_images):
            reconstructed_images[i] = reconstructed_image.get()
        return None, reconstructed_images

    adjoint_ones = adjoint_ones[
        tumor_xmin:tumor_xmax, tumor_ymin:tumor_ymax, tumor_zmin:tumor_zmax
    ]

    # padding for each dimension
    pad_x = (reduction_factor - img_shape[0] % reduction_factor) % reduction_factor
    pad_y = (reduction_factor - img_shape[1] % reduction_factor) % reduction_factor
    pad_z = (reduction_factor - img_shape[2] % reduction_factor) % reduction_factor
    padding = ((0, pad_x), (0, pad_y), (0, pad_z))
    reconstructed_images = [
        xp.pad(image, padding, mode="constant", constant_values=0)
        for image in reconstructed_images
    ]
    padded_shape = reconstructed_images[0].shape
    reduced_img_shape = (
        padded_shape[0] // reduction_factor,
        padded_shape[1] // reduction_factor,
        padded_shape[2] // reduction_factor,
    )
    reconstructed_images = [
        xp.mean(
            reconstructed_image.reshape(
                reduced_img_shape[0],
                reduction_factor,
                reduced_img_shape[1],
                reduction_factor,
                reduced_img_shape[2],
                reduction_factor,
            ),
            axis=(1, 3, 5),
        )
        for reconstructed_image in reconstructed_images
    ]

    decay_map = xp.zeros(reduced_img_shape, dtype=xp.float32)
    reconstructed_activity = xp.zeros(reduced_img_shape, dtype=xp.float32)
    # fill decay map with np.nan
    decay_map.fill(xp.nan)
    reconstructed_activity.fill(xp.nan)

    if fixed_decay_rate is not None:

        def decay_function(x, a):
            return a * np.exp(-fixed_decay_rate * x)

    else:

        def decay_function(x, a, b):
            return a * np.exp(-b * x)

    print("Fitting decay curves...")
    for y_idx in tqdm(range(reduced_img_shape[1])):
        for x_idx in range(reduced_img_shape[0]):
            for z_idx in range(reduced_img_shape[2]):
                if not tumor_mask_dilated[
                    x_idx, y_idx, z_idx
                ]:  # if the voxel is not in the tumor,
                    continue
                voxel_activity = np.array(
                    [
                        reconstructed_images[i][x_idx, y_idx, z_idx].get()
                        for i in range(num_frames)
                    ]
                )
                if fixed_decay_rate is None:
                    try:
                        params, cov = curve_fit(
                            decay_function,
                            frame_times,
                            voxel_activity,
                            p0=[voxel_activity.max(), 1 / frame_times.mean()],
                        )
                        std = np.sqrt(np.diag(cov))
                        if (
                            std[1] > abs(0.2 * params[1])
                            or params[1] < 0
                            or std[0] > 0.2 * params[0]
                            or params[0] < 0
                        ):
                            continue
                    except RuntimeError:
                        continue
                    decay_map[x_idx, y_idx, z_idx] = params[1]
                    reconstructed_activity[x_idx, y_idx, z_idx] = params[0]
                else:
                    try:
                        params, cov = curve_fit(
                            decay_function,
                            frame_times,
                            voxel_activity,
                            p0=[voxel_activity.max()],
                        )
                        std = np.sqrt(np.diag(cov))
                        if std[0] > 0.2 * params[0] or params[0] < 0:
                            continue
                    except RuntimeError:
                        continue
                    decay_map[x_idx, y_idx, z_idx] = fixed_decay_rate
                    reconstructed_activity[x_idx, y_idx, z_idx] = params[0]

    # Removing outliers
    minimum_decay_rate = 1 / maximum_mean_life
    maximum_decay_rate = 1 / minimum_mean_life
    decay_map[decay_map < minimum_decay_rate] = minimum_decay_rate
    decay_map[decay_map > maximum_decay_rate] = maximum_decay_rate
    reconstructed_activity[reconstructed_activity < 0] = 0
    # since there is still information surrounding the tumor, so know everything outside a dilation will be 0
    decay_map[~tumor_mask_dilated] = 0
    reconstructed_activity[~tumor_mask_dilated] = 0
    decay_map = interpolate_nan_values(decay_map)
    reconstructed_activity = interpolate_nan_values(reconstructed_activity)

    xp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    return decay_map, reconstructed_activity


def dynamic_decay_reconstruction_no_fit(
    psf_path,
    img_shape=(248, 140, 176),
    voxel_size=(1.9531, 1.9531, 1.5),
    scanner="vision",
    num_subsets=2,
    osem_iterations=3,
    body_mask=None,  # array with the same shape as the image, 1 for body, 0 for air
    sensitivity_array=None,
    frame_duration=3,  # in minutes
    end_time=30,  # in minutes
    reduction_factor=1,
    tumor_xmin=0,
    tumor_xmax=-1,
    tumor_ymin=0,
    tumor_ymax=-1,
    tumor_zmin=0,
    tumor_zmax=-1,
    tumor_mask_dilated=None,
    patient_info_path=None,
):
    """
    Reconstruction from listmode using parallelproj library. Adding realistic effects to the PET data.
    The output is the reconstructed PET frames.
    """
    # choose a device (CPU or CUDA GPU)
    if "numpy" in xp.__name__:
        # using numpy, device must be cpu
        dev = "cpu"
    elif "cupy" in xp.__name__:
        # using cupy, only cuda devices are possible
        dev = xp.cuda.Device(0)
    elif "torch" in xp.__name__:
        # using torch valid choices are 'cpu' or 'cuda'
        dev = "cuda"

    if scanner == "quadra":
        # Create a Quadra PET scanner
        num_rings = 320
        radius = 410.0  # mm
        max_z = 530.0
        num_sides = 760
        # TOF
        TOF_resolution = 225
        psf = 3.5  # resolution of the scanner in mm
        depth = 20  # mm depth of crystal (depth is the max DOI if all photons are incident perpendicular to the crystal)
        mu = 0.082  # mm^-1 attenuation coefficient
        FWHM_detector = (
            3.39  # mm (given by detector side length, 2*pi*radius/num_sides)
        )
    elif scanner == "vision":
        # Create a Vision PET scanner
        radius = 410.0  # mm
        num_rings = 80
        max_z = 131.15
        num_sides = 760
        # TOF
        TOF_resolution = 225
        psf = 3.5  # resolution of the scanner in mm
        depth = 20  # mm depth of crystal (depth is the max DOI if all photons are incident perpendicular to the crystal)
        mu = 0.082  # mm^-1 attenuation coefficient
        FWHM_detector = (
            3.39  # mm (given by detector side length, 2*pi*radius/num_sides)
        )
    else:
        raise ValueError("Invalid scanner type")

    tofbin_FWHM = (
        TOF_resolution * 1e-12 * 3e8 / 2 * 1e3
    )  # *1e3 to mm;  *1e-12 to s; *3e8 to m/s;  /2 to get one-way distance;
    sigma_tof = tofbin_FWHM / 2.355  # to get sigma from FWHM
    tofbin_width = (
        sigma_tof * 1.03
    )  # sigma_tof * 1.03, as given in https://parallelproj.readthedocs.io/en/stable/python_api.html#module-parallelproj.tof # ps, it is the minimum time difference between the arrival of two photons that it can detect. it is diveded by 2 because if one of them arrivs TOF_resolution

    num_tofbins = 201
    if num_tofbins % 2 == 0:
        num_tofbins -= 1
    print("num_tofbins", num_tofbins)
    tof_params = parallelproj.TOFParameters(
        sigma_tof=sigma_tof, num_tofbins=num_tofbins, tofbin_width=tofbin_width
    )
    enable_tof = True

    # Blurring due to detector resolution, crystal size, DOI, positron range
    res_model = parallelproj.GaussianFilterOperator(
        img_shape, sigma=psf / (2.35 * xp.asarray(voxel_size))
    )

    adjoint_ones = to_device(xp.asarray(sensitivity_array, dtype=xp.float32), dev)

    # Define the structure format for one data record
    format_string = "Q f i f f f f f f h h"
    record_size = struct.calcsize(format_string)

    # Define the dtype for numpy based on the format string
    dtype = np.dtype(
        [
            ("emission_time", "u8"),  # unsigned long long int (Q) (emission_time (ps))
            ("travel_time", "f4"),  # float (f) (travel_time (ps))
            ("emission_voxel", "i4"),  # int (i) (emission voxel)
            ("energy", "f4"),  # float (f) (energy)
            ("z", "f4"),  # float (f) (z (cm))
            ("phi", "f4"),  # float (f) (phi (rad))
            (
                "vx",
                "f4",
            ),  # float (f) (vx; x component of the incident photon direction)
            (
                "vy",
                "f4",
            ),  # float (f) (vy; y component of the incident photon direction)
            (
                "vz",
                "f4",
            ),  # float (f) (vz; z component of the incident photon direction)
            (
                "index1",
                "i2",
            ),  # short int (h)  Flag for scatter: =0 for non-scattered, =1 for Compton, =2 for Rayleigh, and =3 for multiple scatter)
            ("index2", "i2"),  # short int (h) (index2)
        ]
    )

    # Read the binary file in chunks and convert directly to DataFrame
    chunk_size = 100000
    events = []
    with open(psf_path, "rb") as file:
        while True:
            data = file.read(record_size * chunk_size)
            if not data:
                break
            chunk = np.frombuffer(data, dtype=dtype)
            event = pd.DataFrame(chunk).loc[
                :, ["emission_time", "travel_time", "z", "phi", "vx", "vy", "vz"]
            ]
            events.append(event)

    del chunk, event, data
    events = pd.concat(events, ignore_index=True)

    # Dynamic PET: let's split the events into 10 X-minute frames
    start_time = 0
    end_time = end_time * 60 * 1e12  # in picoseconds
    frame_duration_ps = frame_duration * 60 * 1e12  # in picoseconds
    frame_edges = np.arange(start_time, end_time + frame_duration_ps, frame_duration_ps)
    frame_labels = np.arange(len(frame_edges) - 1)
    events["frame"] = pd.cut(
        events["emission_time"], bins=frame_edges, labels=frame_labels, right=False
    )
    num_frames = len(frame_labels)
    if patient_info_path is not None:
        with open(patient_info_path, "a") as patient_info_file:
            patient_info_file.write(f"Number of events: {events.shape[0]}\n")
    if (
        tumor_mask_dilated is not None
    ):  # since there is still information surrounding the tumor, so know everything outside a dilation will be 0
        tumor_mask_dilated = tumor_mask_dilated[
            tumor_xmin:tumor_xmax, tumor_ymin:tumor_ymax, tumor_zmin:tumor_zmax
        ]
    else:
        raise ValueError("tumor_mask_dilated is required")
    reconstructed_images = []
    all_events = events.copy()
    for num_frame in range(num_frames):
        events = all_events[all_events["frame"] == num_frame]
        num_events = events.shape[0]
        travel_time = xp.asarray(events.travel_time)
        vx = xp.asarray(events.vx)
        vy = xp.asarray(events.vy)
        vz = xp.asarray(events.vz)
        phi = xp.asarray(events.phi)
        events_x = radius * xp.cos(phi)
        events_y = radius * xp.sin(phi)
        events_z = xp.asarray(events.z)
        # 1. DOI effect
        # Accounting for the angle of incidence of the photon, larger DOI if incidence angle is larger
        # We are getting the length of the path inside the detectors of each photon, solving:
        # r_salida (posicion donde sale del scanner) = r_vec (posicion donde llega al scanner) + v (direccion incidente) * DOI (escalar); sqrt(r_salida(0)**2 + r_salida(1)**2) = R+20mm (radio del escaner + anchura del cristal, ya que sale en el borde del cristal)

        max_dois = (
            1
            / (vx**2 + vy**2)
            * (
                -(vx * events_x + vy * events_y)
                + xp.sqrt(
                    (vx * events_x + vy * events_y) ** 2
                    - (vx**2 + vy**2)
                    * (events_x**2 + events_y**2 - (radius + depth) ** 2)
                )
            )
        )
        del events

        uniform_rands = xp.random.uniform(0, 1, num_events)  # one sample per event

        # Cumulative distribution function capturing that the probability of
        # a photon being detected is higher when photons enter the crystal
        # and decreases exponentially as they move into the crystal, considering a maximum depth of d and normalizing
        # F(x) = 1 - exp(-mu*x) / (1 - exp(-mu*max_doi))
        # Then, inverse transform sampling to sample values from the distribution
        # Sampling from this inverse function means picking random probabilities and finding the points on the distribution that correspond to those probabilities
        # (inverse of the CDF: F^-1(u) = -ln(1 - u(1 - exp(-mu*d))) / mu;

        dois = -np.log(1 - uniform_rands * (1 - xp.exp(-mu * max_dois))) / mu
        del uniform_rands

        # 2. Detector resolution effect:
        # The detector resolution is modeled as a Gaussian distribution with a FWHM of 3.39 mm
        # Values are sampled from a normal distribution with FWHM = 3.39 mm and added to the x, y, and z coordinates of the events
        sigma_detector = FWHM_detector / 2.355
        mean = 0.0
        event_displacement = xp.random.normal(mean, sigma_detector, num_events * 3)

        events_x = (
            events_x + event_displacement[:num_events] + vx * dois
        )  # x position of the detector
        events_y = (
            events_y + event_displacement[num_events : 2 * num_events] + vy * dois
        )  # y position of the detector
        events_z = (
            events_z * 10.0
            + event_displacement[2 * num_events : 3 * num_events]
            + vz * dois
        )  # *10.0 for cm to mm

        # 3. Crystal size effects:
        # the angle and z position of the event are rounded to the position of the crystal
        crystal_phi_positions = (
            xp.linspace(0, 2 * xp.pi, num_sides, endpoint=False) - np.pi
        )
        events_phi = xp.arctan2(
            events_y, events_x
        )  # angle of the event once DOI and detector resolution are considered
        events_phi[events_phi > xp.pi - 2 * xp.pi / num_sides] -= (
            2 * xp.pi
        )  # wrap around, so that if it wont be assigned to the bin below when it should be assigned to phi=-pi
        events_phi = (
            xp.digitize(events_phi, crystal_phi_positions) * 2 * xp.pi / num_sides
            - xp.pi
        )  # round to the closest crystal phi position
        events_x = radius * xp.cos(
            events_phi
        )  # x position of the crystal corresponding to the event
        events_y = radius * xp.sin(
            events_phi
        )  # y position of the crystal corresponding to the event
        crystal_z_positions = xp.linspace(-max_z, max_z, num_rings)
        events_z = (
            xp.digitize(xp.asarray(events_z), crystal_z_positions)
            * 2
            * max_z
            / num_rings
            - max_z
        )  # round to the closest crystal z position

        # TOF bin
        bin = xp.round(
            (travel_time[0::2] - travel_time[1::2]) / (TOF_resolution / 2.355 * 1.03)
        ).astype(
            int
        )  # / 2.355 * 1.03 to match the spatial tof_bin width
        bin = xp.repeat(bin, 2)
        del (
            crystal_phi_positions,
            crystal_z_positions,
            events_phi,
            event_displacement,
            dois,
            travel_time,
            vx,
            vy,
            vz,
            max_dois,
            phi,
        )

        event_start_coordinates = xp.asarray(
            xp.stack((events_x[0::2], events_y[0::2], events_z[0::2]), axis=1)
        )
        event_end_coordinates = xp.asarray(
            xp.stack((events_x[1::2], events_y[1::2], events_z[1::2]), axis=1)
        )
        event_tof_bins = bin[0::2]
        del events_x, events_y, events_z, bin

        lm_proj = parallelproj.ListmodePETProjector(
            event_start_coordinates, event_end_coordinates, img_shape, voxel_size
        )

        if enable_tof:
            lm_proj.tof_parameters = tof_params
            lm_proj.event_tofbins = event_tof_bins
            lm_proj.tof = enable_tof

        subset_slices = [slice(i, None, num_subsets) for i in range(num_subsets)]

        lm_pet_subset_linop_seq = []

        for i, sl in enumerate(subset_slices):
            subset_lm_proj = parallelproj.ListmodePETProjector(
                event_start_coordinates[sl, :],
                event_end_coordinates[sl, :],
                img_shape,
                voxel_size,
            )

            # enable TOF in the LM projector
            subset_lm_proj.tof_parameters = lm_proj.tof_parameters
            if lm_proj.tof:
                subset_lm_proj.event_tofbins = 1 * event_tof_bins[sl]
                subset_lm_proj.tof = lm_proj.tof

            lm_pet_subset_linop_seq.append(
                parallelproj.CompositeLinearOperator((subset_lm_proj, res_model))
            )

        del event_start_coordinates, event_end_coordinates, event_tof_bins

        lm_pet_subset_linop_seq = parallelproj.LinearOperatorSequence(
            lm_pet_subset_linop_seq
        )

        # number of OSEM iterations
        num_iter = osem_iterations
        beta = 0.0  # 0.05
        # adjoint_ones = adjoint_ones + adjoint_ones.max()
        x = xp.ones(img_shape, dtype=xp.float32, device=dev)

        for i in range(num_iter):
            for k, sl in enumerate(subset_slices):
                print(
                    f"OSEM iteration {(k+1):03} / {(i + 1):03} / {num_iter:03}",
                    end="\r",
                )
                x = lm_em_update(
                    x,
                    lm_pet_subset_linop_seq[k],
                    adjoint_ones / num_subsets,
                )
                x = (1.0 - beta) * x + beta * median_filter(x, size=3)

        # Using the sensitivity as a calibration factor to obtain the activity
        # first, we normalize the image to the number of counts per second, then we dibide by the sensitivity (calibrated) to get the activity
        num_counts = num_events // 2
        x = x / x.sum() * (num_counts / (frame_duration * 60)) / adjoint_ones

        if body_mask is not None:
            x[~body_mask] = 0
        x = x[tumor_xmin:tumor_xmax, tumor_ymin:tumor_ymax, tumor_zmin:tumor_zmax]
        x[~tumor_mask_dilated] = 0
        reconstructed_images.append(x)
        for subset in lm_pet_subset_linop_seq:
            del subset
        del lm_pet_subset_linop_seq, lm_proj, subset_lm_proj

    # padding for each dimension
    pad_x = (reduction_factor - img_shape[0] % reduction_factor) % reduction_factor
    pad_y = (reduction_factor - img_shape[1] % reduction_factor) % reduction_factor
    pad_z = (reduction_factor - img_shape[2] % reduction_factor) % reduction_factor
    padding = ((0, pad_x), (0, pad_y), (0, pad_z))
    reconstructed_images = [
        xp.pad(image, padding, mode="constant", constant_values=0)
        for image in reconstructed_images
    ]
    padded_shape = reconstructed_images[0].shape
    reduced_img_shape = (
        padded_shape[0] // reduction_factor,
        padded_shape[1] // reduction_factor,
        padded_shape[2] // reduction_factor,
    )
    reconstructed_images = [
        xp.mean(
            reconstructed_image.reshape(
                reduced_img_shape[0],
                reduction_factor,
                reduced_img_shape[1],
                reduction_factor,
                reduced_img_shape[2],
                reduction_factor,
            ),
            axis=(1, 3, 5),
        )
        for reconstructed_image in reconstructed_images
    ]

    reconstructed_images = xp.stack(reconstructed_images, axis=0)

    xp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    return reconstructed_images
