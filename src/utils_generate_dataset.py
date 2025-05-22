"""Utility functions for the monte carlo simulations (MCGPU-PET for PET, FRED for proton therapy)
and for image processing"""
import os
import subprocess
import gzip
import numpy as np
import array_api_compat.cupy as xp
import itk
import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import median_filter, zoom


def generate_sensitivity(
    img_shape,
    voxel_sizes,
    CT,
    mcgpu_location,
    mcgpu_input_location,
    sensitivity_location,
    hu2densities_path,
    scan_time,  # in seconds
    factor_activity=1.0,
):
    """Generate the sensitivity array for the MCGPU-PET simulation with a FOV of uniform activity"""
    # voxel sizes in mm (we convert later to cm)
    gen_voxel(
        CT,
        np.ones(shape=img_shape, dtype=np.float32) * factor_activity,
        out_path=os.path.join(mcgpu_location, "phantom.vox"),
        hu2densities_path=hu2densities_path,
        nvox=img_shape,
        dvox=voxel_sizes / 10,
    )  # dvox in cm, change factor multiplying the activity to change the time of the simulation

    # output the image of Trues
    with open(mcgpu_input_location, "r") as file:
        lines = file.readlines()
    psf_keyword = "# REPORT PSF"
    mean_life_keyword = "# ISOTOPE MEAN LIFE"
    scan_time_keyword = "# TOTAL PET SCAN ACQUISITION TIME"
    for idx, line in enumerate((lines)):
        if psf_keyword in line:
            lines[idx] = " 2" + line[2:]
        elif mean_life_keyword in line:
            lines[idx] = (
                " "
                + f"{1000000} "
                + "# ISOTOPE MEAN LIFE\n"  # very large mean life to have a fixed activity
            )
        elif scan_time_keyword in line:
            lines[
                idx
            ] = f"{scan_time:.2f}            # TOTAL PET SCAN ACQUISITION TIME [seconds]\n"
    with open(mcgpu_input_location, "w") as file:
        file.writelines(lines)

    command = ["./MCGPU-PET.x", mcgpu_input_location]
    subprocess.run(command, cwd=mcgpu_location)

    with gzip.open(os.path.join(mcgpu_location, "image_Trues.raw.gz"), "rb") as f:
        sensitivity = (
            xp.frombuffer(f.read(), dtype=xp.int32).reshape(img_shape, order="F").copy()
        )

    # the sensitivity is the total number of counts, and we now dvidide by the activity so that it's a calibration factor
    # from the number of counts to the activity
    sensitivity = sensitivity / (
        xp.ones(shape=img_shape, dtype=np.float32) * factor_activity * scan_time
    )
    sensitivity[
        sensitivity == 0
    ] = 1e-6  # to avoid dividing by zero, only happens at the edge of FOV
    # smooth the sensitivity
    sensitivity = median_filter(sensitivity, size=3)
    np.save(sensitivity_location, sensitivity.get())

    # plot the sensitivity
    plt.figure()
    plt.imshow(sensitivity.get()[:, sensitivity.shape[1] // 2, :], cmap="jet")
    plt.savefig(sensitivity_location[:-4] + ".jpg")

    # output the psf
    with open(mcgpu_input_location, "r") as file:
        lines = file.readlines()
    psf_keyword = "# REPORT PSF"
    for idx, line in enumerate((lines)):
        if psf_keyword in line:
            lines[idx] = " 1" + line[2:]
    with open(mcgpu_input_location, "w") as file:
        file.writelines(lines)

    os.remove(os.path.join(mcgpu_location, "image_Trues.raw.gz"))
    os.remove(os.path.join(mcgpu_location, "image_Scatter.raw.gz"))
    os.remove(os.path.join(mcgpu_location, "sinogram_Trues.raw.gz"))
    os.remove(os.path.join(mcgpu_location, "sinogram_Scatter.raw.gz"))
    return sensitivity


def gen_voxel(
    CT,
    activity_array,
    out_path,
    hu2densities_path,
    nvox=(248, 140, 176),
    dvox=[0.19531, 0.19531, 0.15],
):
    """Generate the material, density and activity arrays for the MCGPU-PET simulation for a given CT image,
    saved into a .vox file.
    """
    # materials = ["air", "lung", "adipose", "water",
    #             "breast_glandular","glands_others", "stomach_intestines",
    #             "muscle", "skin", "spongiosa"]
    densities = np.array([0.0012, 0.3, 0.95, 1.0, 1.02, 1.03, 1.04, 1.05, 1.09, 1.18])
    hu_2_density_map = {}  # Hounsfield unit to density map
    with open(hu2densities_path, "r") as file:
        lines = file.readlines()[2:]  # Skip the first two lines
        for line in lines:
            line_parts = line.split()
            hu = line_parts[1]
            density = line_parts[2]
            hu_2_density_map[int(hu)] = float(density)

    CT = np.round(CT).astype(int)
    hu_2_density_vectorized_mapping = np.vectorize(hu_2_density_map.get)
    density_array = hu_2_density_vectorized_mapping(CT)

    material_array = (
        np.argmin(np.abs(density_array[..., None] - densities), axis=-1) + 1
    )
    material_flat = material_array.flatten(order="F")
    density_flat = density_array.flatten(order="F")
    activity_flat = activity_array.flatten(order="F")

    # removed materials for redundancy: soft tissue, kidneys (1.05), brain(1.05), liver(1.05), cartilage(1.05), eyes (1.05)
    out = open(out_path, "w")

    # -- WRITE HEADER
    out.write("[SECTION VOXELS HEADER v.2008-04-13]\n")
    out.write(
        str(nvox[0])
        + " "
        + str(nvox[1])
        + " "
        + str(nvox[2])
        + "   No. OF VOXELS IN X,Y,Z\n"
    )
    out.write(
        str(dvox[0])
        + " "
        + str(dvox[1])
        + " "
        + str(dvox[2])
        + "   VOXEL SIZE (cm) ALONG X,Y,Z\n"
    )
    out.write(" 1                  COLUMN NUMBER WHERE MATERIAL ID IS LOCATED\n")
    out.write(" 2                  COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED\n")
    out.write(" 0                  BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)\n")
    out.write(
        "[END OF VXH SECTION]  # MCGPU-PET voxel format: Material  Density  Activity\n"
    )

    # Write the flattened arrays as columns separated by space
    for material, density, activity in zip(material_flat, density_flat, activity_flat):
        out.write(f"{material} {density} {activity}\n")

    out.close()
    return None


def crop_save_head_image(
    file_path,
    uncropped_shape=(272, 272, 176),
    xmin=0,
    xmax=None,
    ymin=0,
    ymax=None,
    zmin=0,
    zmax=None,
    is_CT_image=False,
    crop_body=False,
    body_coords=None,
    save_raw=False,
):
    """Read the image from the file and crop it to the desired size"""
    with open(file_path, "rb") as f:
        if is_CT_image:  # Provide the .raw directly, without the mhd header
            img = (
                np.frombuffer(f.read(), dtype=np.int16)
                .reshape(uncropped_shape, order="F")
                .astype(np.float32)
            )
        else:
            # Read the header line by line until we find the line that specifies the binary data
            while True:
                line = f.readline()
                if line.strip() == b"ElementDataFile = LOCAL":
                    break  # Stop after the header line
            # Read the img
            img = np.frombuffer(f.read(), dtype=np.float32).reshape(
                uncropped_shape, order="F"
            )

    if crop_body and body_coords is not None:
        img_full = img.copy()
        img = np.zeros_like(img_full)
        img[body_coords] = img_full[body_coords]

    if xmax is None:
        xmax = uncropped_shape[0]
    if ymax is None:
        ymax = uncropped_shape[1]
    if zmax is None:
        zmax = uncropped_shape[2]
    img = img[xmin:xmax, ymin:ymax, zmin:zmax]

    if save_raw and not is_CT_image:
        img_raw = img.transpose(2, 1, 0)
        img_raw.tofile(file_path[:-3] + "raw")

    if not is_CT_image:
        os.remove(file_path)
    return img


def get_isotope_factors(
    initial_time,
    final_time,
    irradiation_time=0,
    isotope_list=["C11", "N13", "O15"],
    lambda_bio_dict=None,
    component_fraction_dict=None,
):
    """
    Get the factors to convert the number of activated isotopes to activity in each anatomical area
    """
    # Initial and final time of PET measurements in ***minutes****
    # Half lives
    T_1_2_C11 = 20.4  # minutes
    T_1_2_N13 = 9.965  # minutes
    T_1_2_O15 = 2.04  # minutes
    T_1_2_C10 = 0.32  # minutes
    T_1_2_O14 = 1.18  # minutes
    T_1_2_P30 = 2.498  # minutes
    T_1_2_K38 = 7.637  # minutes

    # Decay constants. If biological decay is not considered, only the physical decay is considered
    lambda_dict = {}
    lambda_dict["C11"] = np.log(2) / T_1_2_C11
    lambda_dict["N13"] = np.log(2) / T_1_2_N13
    lambda_dict["O15"] = np.log(2) / T_1_2_O15
    lambda_dict["C10"] = np.log(2) / T_1_2_C10
    lambda_dict["O14"] = np.log(2) / T_1_2_O14
    lambda_dict["P30"] = np.log(2) / T_1_2_P30
    lambda_dict["K38"] = np.log(2) / T_1_2_K38

    # Biological decay constants
    if lambda_bio_dict is None:
        # Medium and fast components are based on Toramatsu et al. 2018, slow based on Parodi et al. 2007
        lambda_bio_dict = {}
        lambda_bio_dict["C11"] = {
            "air": {"fast": 21.04, "medium": 0.3, "slow": 0.0},
            "fat": {"fast": 21.04, "medium": 0.3, "slow": np.log(2) * 60 / 15000},
            "brain": {"fast": 21.04, "medium": 0.3, "slow": np.log(2) * 60 / 10000},
            "soft bone": {"fast": 21.04, "medium": 0.3, "slow": np.log(2) * 60 / 8000},
            "compact bone": {
                "fast": 21.04,
                "medium": 0.3,
                "slow": np.log(2) * 60 / 15000,
            },
        }
        lambda_bio_dict["O15"] = {
            "air": {"fast": 0.0, "medium": 0.72, "slow": 0.024},
            "fat": {"fast": 0.0, "medium": 0.72, "slow": 0.024},
            "brain": {"fast": 0.0, "medium": 0.72, "slow": 0.024},
            "soft bone": {"fast": 0.0, "medium": 0.72, "slow": 0.024},
            "compact bone": {"fast": 0.0, "medium": 0.72, "slow": 0.024},
        }

        # # To remove biological washout uncomment this section and comment the lambda_bio_dict['C11'] and lambda_bio_dict['O15'] above as well as the component_fraction_dict['C11'] and component_fraction_dict['O15'] below
        # lambda_bio_dict['C11'] = {'air': {'fast':0., 'medium':0., 'slow':np.log(2) * 60 /10000},
        #                           'fat': {'fast':0., 'medium':0., 'slow':np.log(2) * 60 /10000},
        #                             'brain': {'fast':0., 'medium':0., 'slow':np.log(2) * 60 /10000},
        #                             'soft bone': {'fast':0., 'medium':0., 'slow':np.log(2) * 60 /10000},
        #                             'compact bone': {'fast':0., 'medium':0., 'slow':np.log(2) * 60 /10000}}
        # lambda_bio_dict['O15'] = lambda_bio_dict['C11']
        # component_fraction_dict = {}  # fraction of slow, medium and fast components in each organ
        # component_fraction_dict['C11'] = {'air': {'fast':0., 'medium':0., 'slow':1.},
        #                                     'fat': {'fast':0., 'medium':0., 'slow':1.},
        #                                     'brain': {'fast':0., 'medium':0., 'slow':1.},
        #                                     'soft bone': {'fast':0., 'medium':0., 'slow':1.},
        #                                     'compact bone': {'fast':0., 'medium':0., 'slow':1}}
        # component_fraction_dict['O15'] = component_fraction_dict['C11']

        lambda_bio_dict["C10"] = lambda_bio_dict["C11"]
        lambda_bio_dict["O14"] = lambda_bio_dict["O15"]
        lambda_bio_dict["N13"] = lambda_bio_dict[
            "O15"
        ]  # For now setting K38 to O15 values
        lambda_bio_dict["P30"] = lambda_bio_dict["O15"]
        lambda_bio_dict["K38"] = lambda_bio_dict[
            "O15"
        ]  # For now setting K38 to O15 values

    if component_fraction_dict is None:
        component_fraction_dict = (
            {}
        )  # fraction of slow, medium and fast components in each organ
        component_fraction_dict["C11"] = {
            "air": {"fast": 0.0, "medium": 0.0, "slow": 1.0},
            "fat": {
                "fast": 0.2 * (1 - 0.9) / 0.52,
                "medium": 0.32 * (1 - 0.9) / 0.52,
                "slow": 0.9,
            },  # * 0.1 / 0.52 is the fraction of medium and fast components taking into account the slow component, which takes precedence
            "brain": {
                "fast": 0.2 * (1 - 0.35) / 0.52,
                "medium": 0.32 * (1 - 0.35) / 0.52,
                "slow": 0.35,
            },
            "soft bone": {
                "fast": 0.2 * (1 - 0.6) / 0.52,
                "medium": 0.32 * (1 - 0.6) / 0.52,
                "slow": 0.6,
            },
            "compact bone": {
                "fast": 0.2 * (1 - 0.9) / 0.52,
                "medium": 0.32 * (1 - 0.9) / 0.52,
                "slow": 0.9,
            },
        }
        component_fraction_dict["O15"] = {
            "air": {"fast": 0.0, "medium": 0.62, "slow": 0.38},
            "fat": {"fast": 0.0, "medium": 0.62, "slow": 0.38},
            "brain": {"fast": 0.0, "medium": 0.62, "slow": 0.38},
            "soft bone": {"fast": 0.0, "medium": 0.62, "slow": 0.38},
            "compact bone": {"fast": 0.0, "medium": 0.62, "slow": 0.38},
        }
        component_fraction_dict["C10"] = component_fraction_dict["C11"]
        component_fraction_dict["O14"] = component_fraction_dict["O15"]
        component_fraction_dict["N13"] = component_fraction_dict["O15"]
        component_fraction_dict["P30"] = component_fraction_dict["O15"]
        component_fraction_dict["K38"] = component_fraction_dict["O15"]

    factor_dict = {}
    for isotope in isotope_list:
        factor_dict[isotope] = {}
        # Taking into account the irradiation time to get the number of activated isotopes, at the end of the irradiation
        if irradiation_time != 0:
            for tissue in lambda_bio_dict[isotope].keys():
                factor_dict[isotope][tissue] = 0
                # Getting the remaining existing radiactive isotopes at the end of the irradiation
                for component in lambda_bio_dict[isotope][tissue].keys():
                    factor_dict[isotope][tissue] += (
                        component_fraction_dict[isotope][tissue][component]
                        / (
                            lambda_bio_dict[isotope][tissue][component]
                            + lambda_dict[isotope]
                        )
                        / irradiation_time
                        * (
                            1
                            - np.exp(
                                -(
                                    lambda_bio_dict[isotope][tissue][component]
                                    + lambda_dict[isotope]
                                )
                                * irradiation_time
                            )
                        )
                    )
            # print(isotope, factor_dict[isotope])
        else:
            N0 = 1
            for tissue in lambda_bio_dict[isotope].keys():
                factor_dict[isotope][tissue] = N0

        for tissue in lambda_bio_dict[isotope].keys():
            # Getting the activated isotopes during the PET measurements
            decay_factor = 0
            for component in lambda_bio_dict[isotope][tissue].keys():
                # # For the factor to reduce the number of activations to match those expected in the PET measurements from inital to final time: (parallelproj)
                # decay_factor += component_fraction_dict[isotope][tissue][component] * (np.exp(-(lambda_dict[isotope] + lambda_bio_dict[isotope][tissue][component]) * initial_time) - np.exp(-(lambda_dict[isotope] + lambda_bio_dict[isotope][tissue][component]) * final_time))
                # # For the factor to reduce the number of activations to match those expected in the PET measurements at the initial time: (parallelproj)
                # decay_factor += component_fraction_dict[isotope][tissue][component] * (np.exp(-(lambda_dict[isotope] + lambda_bio_dict[isotope][tissue][component]) * initial_time))
                # # For the factor to convert the activation into activity at the start of the PET measurements (MCGPU-PET)
                decay_factor += (
                    component_fraction_dict[isotope][tissue][component]
                    * (
                        lambda_dict[isotope]
                        + lambda_bio_dict[isotope][tissue][component]
                    )
                    / 60
                    * np.exp(
                        -(
                            lambda_dict[isotope]
                            + lambda_bio_dict[isotope][tissue][component]
                        )
                        * initial_time
                    )
                )
            factor_dict[isotope][tissue] *= decay_factor
            # print(f"{isotope} in {tissue}: {factor_dict[isotope][tissue]}")
    return factor_dict


def convert_CT_to_mhd(
    mhd_file,
    CT_voxel_size,
    dicom_dir=None,
    matRad_output=None,
    voxel_size=None,
    dev=None,
):  # device for cupy
    """Convert a DICOM (not maintained) or .mat CT file to MHD format"""

    # If the DICOM directory is provided, use it to read the CT image
    if dicom_dir is not None:
        ## NOT MAINTAINED, USING MATRAD OUTPUT INSTEAD, WOULD HAVE TO ADAPT SIZE OF CT CUBE TO DESIRED SIZE
        # Initialize the names generator
        names_generator = itk.GDCMSeriesFileNames.New()
        names_generator.SetDirectory(dicom_dir)

        PixelType = itk.SS
        Dimension = 3

        ImageType = itk.Image[PixelType, Dimension]

        # Get the series UID. Assuming there's only one series in the directory for simplicity.
        series_uid = names_generator.GetSeriesUIDs()
        if not series_uid:
            raise RuntimeError("No DICOM series found in the specified directory.")

        # Use the first series UID to get file names. Modify as needed if handling multiple series.
        file_names = names_generator.GetFileNames(series_uid[0])

        # Initialize and configure the reader
        reader = itk.ImageSeriesReader[ImageType].New()
        reader.SetFileNames(file_names)

        # No need to explicitly set an ImageIO as itk.ImageSeriesReader automatically selects one.

        # Read and then write the image
        reader.Update()
        image = reader.GetOutput()
        direction = np.eye(3)
        image.SetDirection(direction)
        itk.imwrite(image, mhd_file)

    elif matRad_output is not None:
        CT_offset = matRad_output["CT_offset"].T[
            0
        ]  # CT_offset = ct[0, 0][6][0][0][3].T[0]
        CT_cube = matRad_output["CT_cube"].astype(
            np.int16
        )  # CT_cube = ct[0, 0][9][0][0].astype(np.int16)

        CT_cube = CT_cube.transpose(1, 0, 2)  # to load from matlab
        if voxel_size is not None and (voxel_size != CT_voxel_size).any():
            # Reshape the CT cube to the desired voxel size
            print("Reshaping the CT cube to the desired voxel size")
            print(
                f"Original voxel size: {CT_voxel_size}, Desired voxel size: {voxel_size}"
            )
            print(f"Original image size: {CT_cube.shape}")
            CT_cube = xp.round(
                zoom(
                    xp.asarray(CT_cube, device=dev),
                    (CT_voxel_size / voxel_size),
                    order=3,
                )
            ).astype(np.int16)
            print(f"New image size: {CT_cube.shape}")
        elif voxel_size is None:
            voxel_size = CT_voxel_size.copy()
        CT_cube_shape = CT_cube.shape  # Adjusting from MATLAB to Python
        CT_cube = CT_cube.transpose(2, 1, 0)  # to save in fortran order

        with open(mhd_file[:-4] + ".raw", "wb") as CT_file:
            CT_cube.tofile(CT_file)

        # For MHD, need to save it with the MATLAB order
        with open(mhd_file, "wb") as CT_file:
            CT_file.write(
                (
                    f"ObjectType = Image\n"
                    f"NDims = 3\n"
                    f"BinaryData = True\n"
                    f"BinaryDataByteOrderMSB = False\n"
                    f"CompressedData = False\n"
                    f"TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
                    f"Offset = {CT_offset[0]} {CT_offset[1]} {CT_offset[2]}\n"
                    f"CenterOfRotation = 0 0 0\n"
                    f"AnatomicalOrientation = RAI\n"
                    f"ElementSpacing = {voxel_size[1]} {voxel_size[0]} {voxel_size[2]}\n"
                    f"ITK_non_uniform_sampling_deviation = 0.0001\n"
                    f"DimSize = {CT_cube_shape[1]} {CT_cube_shape[0]} {CT_cube_shape[2]}\n"
                    f"ElementType = MET_SHORT\n"
                    f"ElementDataFile = CT.raw\n"
                ).encode()
            )

    print(f"Conversion complete. MHD file saved at: {mhd_file}")
    return CT_cube_shape


def expand_dimension(min_val, max_val, desired_size, dimension_size):
    """Expand a dimension to the desired size by expanding equally on both sides
    except when the expansion would go beyond the dimension size, in which case the expansion is compensated on the other side
    """
    size = max_val - min_val
    dsize = desired_size - size  # required expansion size
    if dsize <= 0:
        dsize = -dsize
        min_val = min_val + dsize // 2
        max_val = min_val + desired_size
        return min_val, max_val
    max_delta_min = min_val  # Maximum possible expansion in the negative direction
    max_delta_max = (
        dimension_size - max_val
    )  # Maximum possible expansion in the positive direction

    delta_min = dsize // 2
    delta_max = (
        dsize - delta_min
    )  # Ensure delta_min + delta_max = dsize (they will be equal if dsize is even)

    delta_min = min(delta_min, max_delta_min)
    delta_max = dsize - delta_min

    if delta_max > max_delta_max:
        delta_max = max_delta_max
        delta_min = dsize - delta_max
        if delta_min > max_delta_min:
            delta_min = max_delta_min
            delta_max = max_delta_max
            # Maximum possible expansion reached

    min_new = min_val - delta_min
    max_new = max_val + delta_max

    return int(min_new), int(max_new)
