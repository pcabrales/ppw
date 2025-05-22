""" This script generates the dataset of PET frames (input) and clean washout parameter maps (output)
to train the model to estimate the maps. This script covers the proton therapy and PET
simulation, as well PET reconstruction for each digital twin patient.
RUN WITH CONDA ENVIRONMENT prototwin-pet-washout (environment.yml,
install for any PC with conda env create -f environment.yml)
"""

import os
import sys
import gc
import uuid
import time
import shutil
import subprocess
import json
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import array_api_compat.cupy as xp
from cupyx.scipy.ndimage import binary_dilation, zoom
from scipy.io import loadmat
from utils_generate_dataset import (
    get_isotope_factors,
    crop_save_head_image,
    gen_voxel,
    convert_CT_to_mhd,
    generate_sensitivity,
    expand_dimension,
)
from utils_parallelproj import dynamic_decay_reconstruction_no_fit
from utils_regions import get_tumor_regions

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
dev = xp.cuda.Device(0)

# ----------------------------------------------------------------------------------------------------------------------------------------
# USER-DEFINED PR0TOTWIN-PET-WASHOUT PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------------------------
#
#   PATIENT DATA AND OUTPUT FOLDERS
DATASET_NUM = 2
SEED_NUMBER = 42
hu2densities_path = os.path.join(script_dir, "../data/ipot-hu2materials.txt")
fredinp_location = os.path.join(script_dir, "../data/washout-fred.inp")
final_shape = (
    64,
    64,
    64,
)  # Final shape of the decay maps; enough to include the tumor (GTV) and some surrounding tissue
#
#   CHOOSING A DOSE VERIFICATION APPROACH
INITIAL_TIME = 15  # minutes time spent before placing the patient in a PET scanner after the final field is delivered
FINAL_TIME = 45  # minutes
IRRADIATION_TIME = 2  # minutes  # time spent delivering the field
FIELD_SETUP_TIME = 2  # minutes  # time spent setting up the field (gantry rotation)
isotope_list = ["C11", "N13", "O15", "K38"]
minimum_mean_lives = {
    "C11": 500
}  # in seconds (simulating a highly vascularized tumor, corresponds to the
# fastest slow component seen in Toramatsu et al 2022 - 0.05 min^-1, which corresponds only to the biological washout)
maximum_mean_lives = {"C11": 1765.0}  # only physical washout, no biological washout
#
# TUMOR REGIONS
MAX_NUM_REGIONS = 4  # maximum number of regions to divide the tumor into
REDUCTION_FACTOR = 1  # factor to reduce the resolution of the image
NUM_REGION_TYPES = 5  # number of region types (not only a map of the decay is saved, but depending on the decay of the region they are binned into different region types)
#
# MONTE CARLO SIMULATION OF THE TREATMENT
N_plans = 75
nprim = 2.8e5  # number of primary particles
variance_reduction = True
maxNumIterations = 10  # Number of times the simulation is repeated (only if variance reduction is True)
stratified_sampling = True
Espread = 0.006  # fractional energy spread (0.6%)
target_dose = 2.18  # Gy  (corresponds to a standard 72 Gy, 33 fractions treatment)
#
# DYNAMIC PET SIMULATION
scanners = ["vision", "quadra"]  # simultaneous calculations for different scanners
mcgpu_location = os.path.join(script_dir, "./pet-simulation-reconstruction/mcgpu-pet")
mcgpu_executable_location = os.path.join(mcgpu_location, "MCGPU-PET.x")
materials_path = os.path.join(mcgpu_location, "materials")
frame_duration = 6  # in minutes
#
# PET RECONSTRUCTION
num_subsets = 1
osem_iterations = 1
patient_names = [
    "HN-CHUM-007",
    "HN-CHUM-010",
    "HN-CHUM-013",
    "HN-CHUM-015",
    "HN-CHUM-017",
    "HN-CHUM-018",
    "HN-CHUM-021",
    "HN-CHUM-022",
    "HN-CHUM-027",
    "HN-CHUM-053",
    "HN-CHUM-055",
    "HN-CHUM-060",
    "LIVER",
]
voxel_size = np.array(
    [1.9531, 1.9531, 1.5]
)  # in mm; refers to the resolution of the images used to train the model from the CHUM dataset,
# if the voxel size of the images are different, the images will be resampled to this resolution
experiment_folder = "experiment_1"  # experiment_1: only slow component of 11C, experiment_2: only slow component of C11 and 13N
# -----------------------------------------------------------------------------------------------------------------------------------------

# Considering all isotopes for obtaining the activity distribution at the start of the PET scan,
# but only a slow component for the PET simulation itself, since the actual biological washout at that time will be
# well fitted by a single exponential decay (slow component of C11)
PET_isotope = "C11"

N_reference = 2e6  # reference number of particles per bixel, not too relevant, will be scaled to the target dose, just needs to be large enough to avoid rounding errors when multiplying by the weights

# For the background, outside the tumor:
# Half lives taking into account the slow component of the biological washout (averaged across tissues)+ the physical decay
# E.g. for C11, Mean_life_efectiva = 1 / (ln (2) / 1223.4 s + ln(2) / 10000)
# for O15, Mean_life_efectiva = 1 / (ln (2) / 2.04 min / 60 sec s + 0.024 min / 60 sec)
average_mean_lives = {
    "C11": 1572.6,
    "N13": 641.3,
    "O15": 164.9,
    "K38": 522.8,
}  # in seconds

component_fraction_dict_tumor = {
    "C11": {
        "region": {
            "fast": 0.2 * (1 - 0.5) / 0.52,
            "medium": 0.32 * (1 - 0.5) / 0.52,
            "slow": 0.5,
        }
    },
    "O15": {
        "region": {
            "fast": 0.0,
            "medium": 0.62,
            "slow": 0.38,
        }
    },
}  # assuming a typical distribution of fast, medium and slow components in the tumor
component_fraction_dict_tumor["N13"] = component_fraction_dict_tumor["O15"]
component_fraction_dict_tumor["K38"] = component_fraction_dict_tumor["O15"]

washout_HU_regions = [
    -np.inf,
    -150,
    -30,
    200,
    1000,
    +np.inf,
]  # According to Parodi et al. 2007

if variance_reduction:
    nprim = nprim // maxNumIterations

HU_regions = [
    -1000,
    -950,
    -120,
    -83,
    -53,
    -23,
    7,
    18,
    80,
    120,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    2995,
    2996,
]  # HU Regions

# Fix the random seed
random.seed(SEED_NUMBER)
np.random.seed(SEED_NUMBER)
xp.random.seed(SEED_NUMBER)
os.environ["PYTHONHASHSEED"] = str(SEED_NUMBER)

# Save raws (not saving them currently because they are too large and we don't need them)
save_raw = False

for patient_name in patient_names:
    patient_folder = os.path.join(
        script_dir, f"../data/{patient_name}"
    )  # Folder to save the numpy arrays for model training and other data
    dataset_folder = os.path.join(
        patient_folder, f"dataset{DATASET_NUM}"
    )  # Folder to save the dataset information

    origin_dataset_folder = None  # os.path.join(patient_folder, "dataset11")
    if patient_name == "LIVER":
        origin_dataset_folder = None

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        # Saving scanner-independent data in the dataset folder
        os.makedirs(os.path.join(dataset_folder, "clean_decay/"))
        os.makedirs(os.path.join(dataset_folder, "regions/"))
        os.makedirs(os.path.join(dataset_folder, "region_types/"))
        os.makedirs(os.path.join(dataset_folder, "fields/"))
        # Saving the fraction of activity with respect to the case without washout
        os.makedirs(os.path.join(dataset_folder, "clean_washout_fraction/"))
        os.makedirs(os.path.join(dataset_folder, "images/"))
        if origin_dataset_folder is not None:
            shutil.copy(
                os.path.join(origin_dataset_folder, "activation_dict.npy"),
                dataset_folder,
            )
            shutil.copy(
                os.path.join(origin_dataset_folder, "scaling_factor.npy"),
                dataset_folder,
            )

    # Path to the DICOM directory
    # dicom_dir = None  # Before: os.path.join('../../HeadPlans/HN-CHUM-018/data/CT)
    CT_mhd_file = os.path.join(patient_folder, "CT.mhd")  # mhd file with the CT
    # Load matRad treatment plan parameters (CURRENTLY ONLY SUPPORTS MATRAD OUTPUT)
    matRad_output = loadmat(os.path.join(patient_folder, "matRad-output.mat"))

    patient_info_path = os.path.join(dataset_folder, "patient_info.txt")
    with open(patient_info_path, "w") as patient_info_file:
        patient_info_file.write("Patient Information\n")
        patient_info_file.write("====================\n")

    scanner_folders = {}
    for scanner in scanners:
        scanner_folder = os.path.join(dataset_folder, scanner)
        scanner_folders[scanner] = scanner_folder
        if not os.path.exists(scanner_folder):
            os.makedirs(scanner_folder)
            os.makedirs(os.path.join(scanner_folder, "noisy_decay/"))
            os.makedirs(os.path.join(scanner_folder, "images/"))
            os.makedirs(os.path.join(scanner_folder, "plans/"))
            os.makedirs(os.path.join(scanner_folder, "timings/"))
            # Saving the fraction of activity with respect to the case without washout  obta
            os.makedirs(os.path.join(scanner_folder, "noisy_washout_fraction/"))
            if origin_dataset_folder is not None:
                shutil.copy(
                    os.path.join(
                        origin_dataset_folder, scanner, f"sensitivity-{scanner}.npy"
                    ),
                    scanner_folder,
                )
                print(
                    os.path.join(
                        origin_dataset_folder, scanner, f"sensitivity-{scanner}.npy"
                    )
                )

        # Move the mcgpu input to the patient folder
        mcgpu_input_location = os.path.join(mcgpu_location, f"MCGPU-PET-{scanner}.in")
        shutil.copy(mcgpu_input_location, scanner_folder)
        mcgpu_input_dataset_location = os.path.join(
            scanner_folder, os.path.basename(mcgpu_input_location)
        )
        with open(mcgpu_input_dataset_location, "r") as file:
            lines = file.readlines()
        keyword_materials = "mcgpu.gz"  # lines specifying the materials include this string because it is the material composition file
        for idx, input_line in enumerate(lines):
            if keyword_materials in input_line:
                lines[idx] = materials_path + input_line
            elif "TOTAL PET SCAN ACQUISITION TIME" in input_line:
                lines[idx] = (
                    f"{(FINAL_TIME - INITIAL_TIME) * 60:.2f}            # TOTAL PET SCAN ACQUISITION TIME [seconds]\n"
                )
        with open(mcgpu_input_dataset_location, "w") as file:
            file.writelines(lines)

    CT_shape = matRad_output["CT_cube"].shape  # Shape of the CT before cropping
    CT_shape = [
        CT_shape[1],
        CT_shape[0],
        CT_shape[2],
    ]  # Adjusting from MATLAB to Python

    CT_voxel_size = matRad_output["CT_resolution"][0]  # in mm
    CT_voxel_size = np.array(
        [CT_voxel_size[1], CT_voxel_size[0], CT_voxel_size[2]]
    )  # Adjusting from MATLAB to Python

    # Convert DICOM to mhd to be processed by FRED, provide matrad_output to remove everything outside the body and avoid the couch interfering with the simulation
    uncropped_shape = convert_CT_to_mhd(
        mhd_file=CT_mhd_file,
        CT_voxel_size=CT_voxel_size,
        matRad_output=matRad_output,
        voxel_size=voxel_size,
        dev=dev,
    )

    L_list = [
        uncropped_shape[0] * voxel_size[0] / 10,
        uncropped_shape[1] * voxel_size[1] / 10,
        uncropped_shape[2] * voxel_size[2] / 10,
    ]  # in cm
    L_line = f"    L=[{', '.join(map(str, L_list))}]"
    activation_line = f"activation: isotopes = [{', '.join(isotope_list)}];"  # activationCode=4TS-747-PSI"  # line introduced in the fred.inp file to score the activation

    with open(hu2densities_path, "r+") as file:
        original_schneider_lines = file.readlines()
    # Replace activation line with the appropriate for the selected isotopes and include variance reduction
    with open(fredinp_location, "r", encoding="utf-8") as file:
        fredinp_lines = file.readlines()
    with open(fredinp_location, "w", encoding="utf-8") as file:
        for line in fredinp_lines:
            if line.lstrip().startswith("L=["):
                line = L_line + "\n"
            elif line.lstrip().startswith("CTscan"):
                line = f"    CTscan={CT_mhd_file}\n"
            elif line.startswith("activation"):
                line = activation_line + "\n"
            elif line.startswith("varianceReduction"):
                # remove the line
                continue
            file.write(line)
        if variance_reduction:
            if stratified_sampling:
                file.writelines(
                    f"varianceReduction: maxNumIterations={maxNumIterations}; lStratifiedSampling=t\n"
                )
            else:
                file.writelines(
                    f"varianceReduction: maxNumIterations={maxNumIterations};\n"
                )

    # Accessing structs
    stf = matRad_output["stf"]
    weights = matRad_output["weights"].T[0]
    isocenter = stf[0, 0][5][0] / 10 - np.array(
        [
            voxel_size[0] / 10 * uncropped_shape[0] / 2,
            voxel_size[1] / 10 * uncropped_shape[1] / 2,
            voxel_size[2] / 10 * uncropped_shape[2] / 2,
        ]
    )  # in cm
    num_fields = stf.shape[1]

    # Finding FWHM for each energy
    machine_data = matRad_output["machine_data"]
    energy_array = []
    FWHM_array = []
    for machine_data_i in range(machine_data.shape[1]):
        energy_array.append(machine_data[0, machine_data_i][1][0][0])
        FWHM_array.append(
            machine_data[0, machine_data_i][7][0][0][2][0][0] / 10
        )  # in cm

    # Finding body mask
    body_indices = matRad_output["body_indices"].T[
        0
    ]  # Before: body_indices = cst[4, 3][0][0].T[0]
    body_indices -= 1  # 0-based indexing, from MATLAB to Python
    body_coords = np.unravel_index(
        body_indices, [CT_shape[2], CT_shape[1], CT_shape[0]]
    )  # Convert to multi-dimensional form
    body_coords = (
        body_coords[1],
        body_coords[2],
        body_coords[0],
    )  # Adjusting from MATLAB to Python
    body_mask = xp.zeros(CT_shape, dtype=np.int16)
    body_mask[body_coords] = 1
    # Resample to the desired voxel size
    body_mask = (
        zoom(xp.asarray(body_mask, device=dev), (CT_voxel_size / voxel_size), order=3)
        > 0.5
    )
    # get body_coords again
    body_coords = np.where(body_mask.get())

    # Get a maximal crop of the body for sensitivity calculation
    indices = xp.where(body_mask)
    xmin, xmax = xp.min(indices[0]).item(), xp.max(indices[0]).item()
    ymin, ymax = xp.min(indices[1]).item(), xp.max(indices[1]).item()
    zmin, zmax = xp.min(indices[2]).item(), xp.max(indices[2]).item()
    with open(patient_info_path, "a") as patient_info_file:
        patient_info_file.write(f"xmin: {xmin}, xmax: {xmax}\n")
        patient_info_file.write(f"ymin: {ymin}, ymax: {ymax}\n")
        patient_info_file.write(f"zmin: {zmin}, zmax: {zmax}\n")

    cropped_shape = (
        -xmin + xmax,
        -ymin + ymax,
        -zmin + zmax,
    )  # Cropped CT including the body, removing empty areas

    body_mask = body_mask[xmin:xmax, ymin:ymax, zmin:zmax]

    # Importing the CTV to find the dose inside it
    CTV_indices = matRad_output["CTV_indices"].T[0]  # Before: cst[32, 3][0][0].T[0]
    CTV_indices -= 1  # 0-based indexing, from MATLAB to Python
    CTV_coords = xp.unravel_index(
        CTV_indices, [CT_shape[2], CT_shape[1], CT_shape[0]]
    )  # Convert to multi-dimensional form
    CTV_coords = (
        CTV_coords[1],
        CTV_coords[2],
        CTV_coords[0],
    )  # Adjusting from MATLAB to Python
    CTV_mask = np.zeros(CT_shape, dtype=np.int16)
    CTV_mask[CTV_coords] = 1
    # Resample to the desired voxel size
    CTV_mask = (
        zoom(xp.asarray(CTV_mask, device=dev), (CT_voxel_size / voxel_size), order=3)
        > 0.5
    )

    CTV_mask = CTV_mask[xmin:xmax, ymin:ymax, zmin:zmax]  # Crop to the body

    # WASHOUT CURVE
    tumor_indices = matRad_output["tumor_indices"].T[0]  # Before: cst[32, 3][0][0].T[0]
    tumor_indices -= 1  # 0-based indexing, from MATLAB to Python
    tumor_coords = xp.unravel_index(
        tumor_indices, [CT_shape[2], CT_shape[1], CT_shape[0]]
    )  # Convert to multi-dimensional form
    tumor_coords = (
        tumor_coords[1],
        tumor_coords[2],
        tumor_coords[0],
    )  # Adjusting from MATLAB to Python

    tumor_mask = xp.zeros(CT_shape, dtype=np.int16)
    tumor_mask[tumor_coords] = 1
    # Resample to the desired voxel size
    tumor_mask = (
        zoom(xp.asarray(tumor_mask, device=dev), (CT_voxel_size / voxel_size), order=3)
        > 0.5
    )

    tumor_mask = tumor_mask[xmin:xmax, ymin:ymax, zmin:zmax]  # Crop to the body
    tumor_mask_dilated = binary_dilation(
        tumor_mask, iterations=6, brute_force=True  # 4 iterations before
    )  # Dilate the tumor mask to include the surrounding tissue

    # Crop regions_mask to where the tumor
    tumor_xmin, tumor_xmax = [
        int(x) for x in xp.where(xp.any(tumor_mask, axis=(1, 2)))[0][[0, -1]]
    ]
    tumor_ymin, tumor_ymax = [
        int(x) for x in xp.where(xp.any(tumor_mask, axis=(0, 2)))[0][[0, -1]]
    ]
    tumor_zmin, tumor_zmax = [
        int(x) for x in xp.where(xp.any(tumor_mask, axis=(0, 1)))[0][[0, -1]]
    ]

    # expand each dimension to get the final_shape
    tumor_xmin, tumor_xmax = expand_dimension(
        tumor_xmin, tumor_xmax, final_shape[0], tumor_mask.shape[0]
    )
    tumor_ymin, tumor_ymax = expand_dimension(
        tumor_ymin, tumor_ymax, final_shape[1], tumor_mask.shape[1]
    )
    tumor_zmin, tumor_zmax = expand_dimension(
        tumor_zmin, tumor_zmax, final_shape[2], tumor_mask.shape[2]
    )

    # save cropped tumor mask
    cropped_tumor_mask = tumor_mask[
        tumor_xmin:tumor_xmax, tumor_ymin:tumor_ymax, tumor_zmin:tumor_zmax
    ]
    np.save(os.path.join(dataset_folder, "tumor_mask.npy"), cropped_tumor_mask)
    cropped_dilated_tumor_mask = tumor_mask_dilated[
        tumor_xmin:tumor_xmax, tumor_ymin:tumor_ymax, tumor_zmin:tumor_zmax
    ]

    # For plotting the tumor mask
    tumor_mask_slice = cropped_tumor_mask[:, cropped_tumor_mask.shape[1] // 2, :].T
    tumor_outline = tumor_mask_slice ^ binary_dilation(
        tumor_mask_slice, iterations=2, brute_force=True
    )
    outline_masked = np.ma.masked_where(tumor_outline.get() == 0, tumor_outline.get())

    # Cropping the CT
    CT_file_path = os.path.join(patient_folder, "CT.raw")
    # CT_cropped HAS THE SHAPE OF THE CT CROPPED TO INCLUDE THE ENTIRE BODY, BUT THE FINAL CT USED
    # FOR THE SIMULATION IS CROPPED TO THE FINAL SHAPE, ONLY INCLUDING THE AREAS WHERE ACTIVITY IS PRESENT
    # SO THE CT SAVED AT CT_npy_path IS MORE CROPPED THAN CT_cropped however ironic it is
    CT_cropped = crop_save_head_image(
        CT_file_path,
        is_CT_image=True,
        uncropped_shape=uncropped_shape,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
    )  # I need to save CT because I use it later
    np.save(os.path.join(patient_folder, "CT_cropped.npy"), CT_cropped)
    CT_npy_path = os.path.join(patient_folder, "CT.npy")

    CT_tumor_cropped = CT_cropped[
        tumor_xmin:tumor_xmax, tumor_ymin:tumor_ymax, tumor_zmin:tumor_zmax
    ]  # Crop to the tumor region
    np.save(CT_npy_path, CT_tumor_cropped)

    # Plotting the CT
    plt.figure()
    plt.imshow(CT_cropped[:, CT_cropped.shape[1] // 2, :].T, cmap="gray")
    plt.title("CT slice")
    plt.savefig(os.path.join(dataset_folder, "images/CT.png"))

    shutil.copy(CT_file_path, dataset_folder)

    # os.remove(CT_file_path)  # For FRED v 3.6

    scanner_sensitivity_arrays = {}
    for scanner in scanners:
        scanner_folder = scanner_folders[scanner]
        # Generate the sensitivity for the reconstruction
        sensitivity_location = os.path.join(
            scanner_folder, f"sensitivity-{scanner}.npy"
        )
        if os.path.exists(sensitivity_location):
            sensitivity_array = np.load(
                sensitivity_location
            )  # check if sensitivity array already exists
        else:
            mcgpu_input_location = os.path.join(
                mcgpu_location, f"MCGPU-PET-{scanner}.in"
            )
            mcgpu_input_dataset_location = os.path.join(
                scanner_folder, os.path.basename(mcgpu_input_location)
            )
            scan_time = 30 * 60  # seconds
            sensitivity_array = generate_sensitivity(
                cropped_shape,
                voxel_size,
                CT_cropped,
                mcgpu_location,
                mcgpu_input_dataset_location,
                sensitivity_location,
                hu2densities_path,
                scan_time,
                factor_activity=2.0,
            )
        scanner_sensitivity_arrays[scanner] = sensitivity_array

    # MCGPU-PET effective isotope mean lives (not half-lives)
    # Half lives taking into account the slow component of the biological washout (averaged across tissues) + the physical decay
    # E.g. for C11, Mean_life_efectiva = 1 / (ln (2) / 1223.4 s + ln(2) / 10000)
    # for O15, Mean_life_efectiva = 1 / (ln (2) / 2.04 min / 60 sec s + 0.024 min / 60 sec)

    distance_target = 8  # cm distance from the target to the begin of the ray; if too large, it will be out of bounds for the FRED simulation
    if patient_name == "HN-CHUM-027":
        distance_target = 5  #  # cm distance from the target to the begin of the ray
    elif patient_name == "HN-CHUM-043":
        distance_target = 3  # cm distance from the target to the begin of the ray
    elif patient_name == "HN-CHUM-046":
        distance_target = 6  # cm distance from the target to the begin of the ray
    elif patient_name == "HN-CHUM-055":
        distance_target = 5
    elif patient_name == "HN-CHUM-060":
        distance_target = 6
    elif patient_name == "LIVER":
        distance_target = 20

    if not os.path.exists(
        os.path.join(dataset_folder, "activation_dict.npy")
    ) or not os.path.exists(os.path.join(dataset_folder, "scaling_factor.npy")):
        delta_x = 0
        delta_y = 0
        delta_psi = 0
        HU_regions_deviations = [0.0 for k in range(len(HU_regions) - 1)]

        # Introduction of deviations in the HU regions
        HU_region = 0
        deviations = HU_regions_deviations[HU_region]
        schneider_lines = original_schneider_lines.copy()
        for line_idx, line in enumerate(schneider_lines):
            CTHU = line_idx - 1002
            if CTHU >= HU_regions[HU_region + 1]:
                HU_region += 1
                deviations = HU_regions_deviations[HU_region]
            line_list = line.strip().split(" ")
            if line_idx >= 2:
                values = np.array(line_list[5:]).astype(float)
                values += values * deviations
                values[1:] /= values[1:].sum() / 100
                line_list[5:] = values.astype(str)
            schneider_lines[line_idx] = " ".join(line_list) + "\n"

        # Rotation matrix
        R = np.array(
            [
                [np.cos(delta_psi), 0, -np.sin(delta_psi)],
                [0, 1, 0],
                [np.sin(delta_psi), 0, np.cos(delta_psi)],
            ]
        )

        # Iterating over the fields
        plan_pb_num = 0  # to keep track of all bixels, or pencil beams (pb) in the plan
        activation_dict = {
            f"field{field_num}": {isotope: 0 for isotope in isotope_list}
            for field_num in range(num_fields)
        }  # to store the activation for each field and isotope
        total_dose = 0
        for field_num in range(num_fields):
            print(f"\nField {field_num} / {num_fields}")
            field_pb_num = (
                0  # to keep track of all bixels, or pencil beams (pb) in the field
            )
            pencil_beams = []  # to store all field pencil beams
            sourcePoint_field = stf[0, field_num][9][0] / 10 + isocenter  # in cm
            field = stf[0, field_num][7][0]
            plan_folder_location = os.path.join(
                dataset_folder, f"fields/field{field_num}"
            )
            os.makedirs(plan_folder_location, exist_ok=True)
            fredinp_destination = os.path.join(
                plan_folder_location, "fred.inp"
            )  # copy fred.inp intro new folder
            shutil.copy(fredinp_location, fredinp_destination)
            for bixel_num, bixel in enumerate(field):
                pos_target = (
                    bixel[2][0] / 10
                )  # in cm, MatRad gives it relative to the isocenter
                # Rotate target
                pos_target_rotated = R @ pos_target
                # Displace target
                pos_target_deviated = [
                    pos_target_rotated[0] + delta_x,
                    pos_target_rotated[1],
                    pos_target_rotated[2] + delta_y,
                ] + isocenter  # delta_y is added in the third dimension because we call y the superior inferior direction (head to feet), but the actual coordinate system is LPS (left-right, posterior-anterior, superior-inferior)
                pb_direction = pos_target_deviated - sourcePoint_field
                pb_direction = pb_direction / np.linalg.norm(pb_direction)
                sourcePoint_bixel = (
                    pos_target_deviated - pb_direction * distance_target
                )  # distance_target cm from target to get out of the body

                for pb_energy in bixel[4][0]:
                    idx_closest = min(
                        range(len(energy_array)),
                        key=lambda energy_val: abs(
                            energy_array[energy_val] - pb_energy
                        ),
                    )  # find closest energy to bixel energy
                    FWHM = FWHM_array[idx_closest]  # get FWHM for that energy
                    pencil_beam_line = (
                        f"pb: {field_pb_num} Phantom; particle = proton; T = {pb_energy}; Espread={Espread}; v={str(list(pb_direction))}; P={str(list(sourcePoint_bixel))};"
                        f"Xsec = gauss; FWHMx={FWHM}; FWHMy={FWHM}; nprim={nprim:.0f}; N={N_reference*weights[plan_pb_num]:.0f};"
                    )
                    field_pb_num += 1
                    plan_pb_num += 1
                    pencil_beams.append(pencil_beam_line)

            with open(fredinp_destination, "a", encoding="utf-8") as file:
                file.write("\n".join(pencil_beams))
                file.write("\n")
                file.writelines(schneider_lines)

            # Execute fred
            command = ["fred"]
            subprocess.run(command, cwd=plan_folder_location)

            # Crop and delete larger files
            # mhd_folder_path = os.path.join(plan_folder_location, "out/reg/Phantom")  # For FRED v 3.6
            mhd_folder_path = os.path.join(
                plan_folder_location, "out/score"
            )  # For FRED v 3.7

            # Isotopes
            for isotope in isotope_list:
                # isotope_file_path = os.path.join(mhd_folder_path, f'{isotope}_scorer.mhd')  # For FRED v 3.6
                isotope_file_path = os.path.join(
                    mhd_folder_path, f"Phantom.Activation_{isotope}.mhd"
                )  # For FRED v 3.7
                activation = crop_save_head_image(
                    isotope_file_path,
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                    zmin=zmin,
                    zmax=zmax,
                    uncropped_shape=uncropped_shape,
                    save_raw=save_raw,
                    crop_body=True,
                    body_coords=body_coords,
                )
                activation_dict[f"field{field_num}"][isotope] = activation

            # Dose
            # dose_file_path = os.path.join(mhd_folder_path, 'Dose.mhd')  # For FRED v 3.6
            dose_file_path = os.path.join(
                mhd_folder_path, "Phantom.Dose.mhd"
            )  # For FRED v 3.7
            total_dose += crop_save_head_image(
                dose_file_path,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                zmin=zmin,
                zmax=zmax,
                uncropped_shape=uncropped_shape,
                crop_body=True,
                body_coords=body_coords,
                save_raw=save_raw,
            )

            # Plotting the dose and activation
            tumor_mask_slice_uncropped = tumor_mask[:, tumor_mask.shape[1] // 2, :].T
            tumor_outline_uncropped = tumor_mask_slice_uncropped ^ binary_dilation(
                tumor_mask_slice_uncropped, iterations=2, brute_force=True
            )
            outline_masked_uncropped = np.ma.masked_where(
                tumor_outline_uncropped.get() == 0, tumor_outline_uncropped.get()
            )

            # ctv_slice_uncropped = CTV_mask[:, CTV_mask.shape[1] // 2, :].T
            # ctv_outline_uncropped = ctv_slice_uncropped ^ binary_dilation(ctv_slice_uncropped, iterations=2, brute_force=True)
            # ctv_outline_masked_uncropped = np.ma.masked_where(ctv_outline_uncropped.get() == 0, ctv_outline_uncropped.get())

            # Plot the dose and activation
            plt.figure()
            plt.imshow(total_dose[:, total_dose.shape[1] // 2, :].T)
            # plot the ctv as an outline
            plt.imshow(
                outline_masked_uncropped, alpha=1.0, vmin=0, vmax=1.0, cmap="cool"
            )
            plt.colorbar()
            plt.savefig(
                os.path.join(dataset_folder, f"images/dose_after_field{field_num}.jpg")
            )

            plt.figure()
            plt.imshow(
                activation_dict[f"field{field_num}"][PET_isotope][
                    :,
                    activation_dict[f"field{field_num}"][PET_isotope].shape[1] // 2,
                    :,
                ].T
            )
            plt.colorbar()
            plt.savefig(
                os.path.join(dataset_folder, f"images/activation_field{field_num}.jpg")
            )
            #

        # Save activation_dict
        np.save(os.path.join(dataset_folder, "activation_dict.npy"), activation_dict)

        # Scaling the dose to the target dose
        # this is done by matching the median dose in the CTV to the target dose (as found acceptable in https://doi.org/10.1186/s13014-022-02143-x)
        if patient_name == "LIVER":
            CTV_mask = (
                tumor_mask.copy()
            )  # For the CORT dataset's liver cancer patient, CTV is too large (whole liver), taking GTV instead
        total_dose_CTV = total_dose[CTV_mask.get()]
        scaling_factor = target_dose / np.median(total_dose_CTV)

        # Save the scaling factor
        np.save(os.path.join(dataset_folder, "scaling_factor.npy"), scaling_factor)

    else:
        activation_dict = np.load(
            os.path.join(dataset_folder, "activation_dict.npy"), allow_pickle=True
        ).item()
        scaling_factor = np.load(os.path.join(dataset_folder, "scaling_factor.npy"))

    print(
        f"Scaling factor to match target dose to median dose in the tumor: {scaling_factor}"
    )
    with open(patient_info_path, "a") as patient_info_file:
        patient_info_file.write(f"Scaling factor: {scaling_factor}\n")
    # Record time for each step
    times_mcgpu = []
    times_region_generation = []
    times_decay_reconstruction = []
    times_iteration = []
    timings_dir = os.path.join(scanner_folder, "timings")
    FILE_ID = uuid.uuid4().hex
    if not os.path.exists(timings_dir):
        os.makedirs(timings_dir)

    # Now, we simulate the PET acquisition for the tumor with different regions or heterogeneities

    # ONLY C11: Get the mean lives for each region type
    edges_region_types = {
        PET_isotope: xp.linspace(
            minimum_mean_lives[PET_isotope],
            maximum_mean_lives[PET_isotope],
            NUM_REGION_TYPES + 1,
        )
    }
    # Save edges_region_types to a txt file
    # Since the regions are saved based on mean life but then we fit decay rates (1/mean_life),
    # the spacing of decay rates won't be uniform as for the mean lives,
    # but it will be larger for lower mean lives, which is actually good because they are harder
    # to distinguish in the PET image and we are using larger bins and because they are less common region types
    edges_region_types_path = os.path.join(dataset_folder, "edges_region_types.txt")
    with open(edges_region_types_path, "w") as file:
        for isotope, edges in edges_region_types.items():
            file.write(f"{isotope}:\n")
            file.write(str(edges) + " s\n")
            file.write(str(1 / edges) + " (1/s)\n")

            for region_num in range(len(edges) - 1):
                file.write(
                    f"Region type {region_num + 1} starts at {edges[region_num]}s and ends at {edges[region_num + 1]}s \n"
                )
                file.write(
                    f"Region type {region_num + 1} starts at {1/edges[region_num]:.2e} (1/s) and ends at {1/edges[region_num + 1]:.2e} (1/s) \n"
                )

    plan_start = 0
    for plan_num in range(plan_start, plan_start + N_plans):
        print(f"\nPatient {patient_name}: Plan {plan_num} / {plan_start + N_plans}")
        start_times_iteration = time.time()

        # Let's make different regions and assign corresponding mean lives. For example, let's set the tumor to be necrotic and thus have no biological washout
        start_time_region_generation = time.time()
        (
            regions,
            mean_lives_regions,
            clean_decay_map,
            region_types_map,
        ) = get_tumor_regions(
            tumor_mask,
            minimum_mean_lives,
            maximum_mean_lives,
            average_mean_lives,
            edges_region_types,
            voxel_size=voxel_size,
            max_num_regions=MAX_NUM_REGIONS,
            isotope_list=[PET_isotope],  # ONLY C11
        )
        clean_decay_map = clean_decay_map[PET_isotope]  # ONLY C11
        clean_decay_map = clean_decay_map[
            tumor_xmin:tumor_xmax, tumor_ymin:tumor_ymax, tumor_zmin:tumor_zmax
        ]
        # # Uncomment to see the regions
        # plt.figure()
        # lower_bound = 1 / maximum_mean_lives[PET_isotope]
        # upper_bound = 1 / minimum_mean_lives[PET_isotope]
        # total_region = np.zeros_like(clean_decay_map.get())
        # for region_num, region in enumerate(regions):
        #     total_region += region[tumor_xmin:tumor_xmax, tumor_ymin:tumor_ymax, tumor_zmin:tumor_zmax] * (region_num + 1)
        # # plt.colorbar()
        # # plt.imshow(clean_decay_map[:, clean_decay_map.shape[1] // 2, :].get(), vmin=lower_bound, vmax=upper_bound)
        # plt.imshow(total_region[:, total_region.shape[1] // 2, :], vmin=0, vmax=len(regions))
        # plt.colorbar()
        # plt.savefig(os.path.join(scanner_folder, f"images/plan{plan_num}_regions.jpg"))
        # #

        clean_decay_map[~cropped_dilated_tumor_mask.get()] = (
            0  # Setting the decay rate outside the body to zero
        )

        np.save(
            os.path.join(dataset_folder, f"clean_decay/plan{plan_num}.npy"),
            clean_decay_map,
        )

        region_types_map = region_types_map[PET_isotope]  # ONLY C11 FOR NOW
        region_types_map = region_types_map[
            tumor_xmin:tumor_xmax, tumor_ymin:tumor_ymax, tumor_zmin:tumor_zmax
        ]  # Crop to the tumor region
        np.save(
            os.path.join(dataset_folder, f"region_types/plan{plan_num}.npy"),
            region_types_map,
        )

        times_region_generation.append(time.time() - start_time_region_generation)

        activity_isotope_dict = {
            isotope: 0 for isotope in isotope_list
        }  # to store the activity for each isotope
        no_washout_activity_isotope_dict = {
            isotope: 0 for isotope in isotope_list
        }  # to store the activity for each isotope without washout

        # based on Toramatsu et al 2022, where they observed up to 4x differences in medium components between specimens
        # for each region, the typical washout is scaled by a random factor between 0.5 and 1.5 to introduce variability
        # this factor is considered the same for all isotopes and components, assuming that, for example, more perfused regions will affect all isotopes in the same way
        random_factor_regions = np.random.uniform(0.5, 1.5, len(regions))
        for field_num in range(num_fields):
            remaining_fields = num_fields - field_num - 1
            field_initial_time = (
                INITIAL_TIME + (FIELD_SETUP_TIME + IRRADIATION_TIME) * remaining_fields
            )  # taking into account the time spent setting up the other fields and delivering them
            field_final_time = (
                FINAL_TIME + (FIELD_SETUP_TIME + IRRADIATION_TIME) * remaining_fields
            )
            field_factor_dict = get_isotope_factors(
                field_initial_time,
                field_final_time,
                IRRADIATION_TIME,
                isotope_list=isotope_list,
            )  # factors to multiply by the activation (N0) to get the number of decays in the given interval

            lambda_bio_dict_no_washout = {}
            component_fraction_dict_no_washout = {}
            for isotope in isotope_list:
                lambda_bio_dict_no_washout[isotope] = {
                    "region": {"fast": 0, "medium": 0, "slow": 0}
                }
                component_fraction_dict_no_washout[isotope] = {
                    "region": {"fast": 0, "medium": 0, "slow": 1}
                }
                for tissue_num, tissue in enumerate(field_factor_dict[isotope].keys()):
                    lambda_bio_dict_no_washout[isotope][tissue] = {
                        "fast": 0,
                        "medium": 0,
                        "slow": 0,
                    }
                    component_fraction_dict_no_washout[isotope][tissue] = {
                        "fast": 0,
                        "medium": 0,
                        "slow": 1,
                    }
            field_factor_dict_no_washout = get_isotope_factors(
                field_initial_time,
                field_final_time,
                IRRADIATION_TIME,
                isotope_list=isotope_list,
                lambda_bio_dict=lambda_bio_dict_no_washout,
                component_fraction_dict=component_fraction_dict_no_washout,
            )

            for isotope in isotope_list:
                for tissue_num, tissue in enumerate(field_factor_dict[isotope].keys()):
                    tissue_mask = (
                        (CT_cropped >= washout_HU_regions[tissue_num])
                        & (CT_cropped < washout_HU_regions[tissue_num + 1])
                        & ~tumor_mask.get()
                    )  # outside the tumor because inside we have the tumor regions
                    activation_tissue = (
                        activation_dict[f"field{field_num}"][isotope].copy()
                        * scaling_factor
                    )
                    activation_tissue[tissue_mask] *= field_factor_dict[isotope][tissue]
                    activation_tissue[~tissue_mask] = 0
                    activity_isotope_dict[isotope] += activation_tissue

                    # No washout activity
                    activation_tissue = (
                        activation_dict[f"field{field_num}"][isotope].copy()
                        * scaling_factor
                    )
                    activation_tissue[tissue_mask] *= field_factor_dict_no_washout[
                        isotope
                    ][tissue]
                    activation_tissue[~tissue_mask] = 0
                    no_washout_activity_isotope_dict[isotope] += activation_tissue

                # iterate over regions to introduce variability in the washout, except for the last region, which is the outside of the tumor
                for region_num, (region, random_factor_region) in enumerate(
                    zip(regions[:-1], random_factor_regions[:-1])
                ):
                    activation_tissue = activation_dict[f"field{field_num}"][
                        isotope
                    ].copy()
                    mean_life_bio = 1 / (
                        1 / mean_lives_regions[PET_isotope][region_num].item()
                        - 1 / maximum_mean_lives[PET_isotope]
                    )  # biological mean life (1 / mean_life_bio = 1 / mean_life_region - 1 / maximum_mean_life)
                    lambda_bio_dict = {
                        "C11": {
                            "region": {
                                "fast": 21.04 * random_factor_region,
                                "medium": 0.3 * random_factor_region,
                                "slow": np.log(2) * 60 / 10000 * random_factor_region,
                            }
                        },
                        "O15": {
                            "region": {
                                "fast": 0.0,
                                "medium": 0.73 * random_factor_region,
                                "slow": 0.024 * random_factor_region,
                            }
                        },
                    }
                    lambda_bio_dict["N13"] = lambda_bio_dict["O15"]
                    lambda_bio_dict["K38"] = lambda_bio_dict["O15"]

                    # for the rest of values, random values are assigned, but for the slow component, the previously
                    # found random mean life, relevant to the PET simulation, is used
                    lambda_bio_dict[PET_isotope]["region"]["slow"] = 1 / (
                        mean_life_bio / 60
                    )  # in 1/min

                    field_factor_dict_region = get_isotope_factors(
                        field_initial_time,
                        field_final_time,
                        IRRADIATION_TIME,
                        isotope_list=isotope_list,
                        component_fraction_dict=component_fraction_dict_tumor,
                        lambda_bio_dict=lambda_bio_dict,
                    )  # factors to multiply by the activation (N0) to get the number of decays in the given interval
                    activation_tissue[region] *= field_factor_dict_region[isotope][
                        "region"
                    ]
                    activation_tissue[~region] = 0
                    activity_isotope_dict[isotope] += activation_tissue

                    # No washout activity
                    activation_tissue = activation_dict[f"field{field_num}"][
                        isotope
                    ].copy()
                    activation_tissue[region] *= field_factor_dict_no_washout[isotope][
                        "region"
                    ]
                    activation_tissue[~region] = 0
                    no_washout_activity_isotope_dict[isotope] += activation_tissue

        PET_isotope_activity = activity_isotope_dict[PET_isotope].copy()
        PET_isotope_activity_no_washout = no_washout_activity_isotope_dict[
            PET_isotope
        ].copy()  # Crop to the tumor region
        for isotope in isotope_list:
            if isotope != PET_isotope:
                PET_isotope_activity += activity_isotope_dict[isotope]
                PET_isotope_activity_no_washout += no_washout_activity_isotope_dict[
                    isotope
                ]

        clean_washout_fraction = PET_isotope_activity / (
            PET_isotope_activity_no_washout + 1e-12
        )
        clean_washout_fraction = clean_washout_fraction[
            tumor_xmin:tumor_xmax, tumor_ymin:tumor_ymax, tumor_zmin:tumor_zmax
        ]  # Crop to the tumor region
        clean_washout_fraction[~cropped_dilated_tumor_mask.get()] = (
            0  # Setting the fraction outside the tumor to zero
        )

        np.save(
            os.path.join(dataset_folder, f"clean_washout_fraction/plan{plan_num}.npy"),
            clean_washout_fraction,
        )

        for scanner in scanners:
            scanner_folder = scanner_folders[scanner]
            sensitivity_array = scanner_sensitivity_arrays[scanner]
            mcgpu_input_location = os.path.join(
                mcgpu_location, f"MCGPU-PET-{scanner}.in"
            )
            mcgpu_input_dataset_location = os.path.join(
                scanner_folder, os.path.basename(mcgpu_input_location)
            )
            # Create folder for each new deviated plan,
            plan_i_location = os.path.join(scanner_folder, f"plans/plan{plan_num}")
            if not os.path.exists(plan_i_location):
                os.makedirs(plan_i_location)
            merged_raw_file = os.path.join(plan_i_location, "merged_MCGPU_PET.psf.raw")
            with open(merged_raw_file, "wb") as merged_file:
                pass  # Create an empty file to start with

            # MCGPU-PET simulation

            start_time_mcgpu = time.time()
            for region_num, (region, mean_life_region) in enumerate(
                zip(regions, mean_lives_regions[PET_isotope])
            ):
                activity_isotope_region = PET_isotope_activity.copy()
                activity_isotope_region[~region] = (
                    0  # Setting the activity outside the region to zero
                )

                out_path = os.path.join(plan_i_location, "phantom.vox")
                gen_voxel(
                    CT_cropped,
                    activity_isotope_region,
                    out_path,
                    hu2densities_path,
                    nvox=cropped_shape,
                    dvox=voxel_size / 10,
                )  # dvox in cm
                shutil.copy(
                    mcgpu_input_dataset_location, plan_i_location
                )  # copy the input file
                os.rename(
                    os.path.join(
                        plan_i_location, os.path.basename(mcgpu_input_location)
                    ),
                    os.path.join(plan_i_location, "MCGPU-PET.in"),
                )  # renaming the input file from whatever name it had

                # Modify the input file to include the isotope's mean life
                input_path = os.path.join(plan_i_location, "MCGPU-PET.in")
                with open(input_path, "r") as file:
                    lines = file.readlines()
                MEAN_LIFE_KEYWORD = "# ISOTOPE MEAN LIFE"
                for idx, input_line in enumerate(lines):
                    if MEAN_LIFE_KEYWORD in input_line:
                        lines[idx] = (
                            " " + f"{mean_life_region} " + "# ISOTOPE MEAN LIFE\n"
                        )
                with open(input_path, "w") as file:
                    file.writelines(lines)

                # running mcgpu
                command = [mcgpu_executable_location, "MCGPU-PET.in"]
                subprocess.run(command, cwd=plan_i_location)

                # Writing the raw file to a single file with the detections of all isotopes
                with open(merged_raw_file, "ab") as merged_file:
                    with open(
                        os.path.join(plan_i_location, "MCGPU_PET.psf.raw"), "rb"
                    ) as isotope_file:
                        merged_file.write(isotope_file.read())

                # remove base folder plan_i_location with all files
                os.remove(os.path.join(plan_i_location, "MCGPU_PET.psf.raw"))
                os.remove(os.path.join(plan_i_location, "phantom.vox"))
                os.remove(os.path.join(plan_i_location, "Energy_Sinogram_Spectrum.dat"))
                os.remove(os.path.join(plan_i_location, "MCGPU_PET.psf"))

            shutil.copy(
                merged_raw_file,
                os.path.join(
                    script_dir,
                    f"pet-simulation-reconstruction/mcgpu-pet/washout_curve/{experiment_folder}/MCGPU_PET.psf.raw",
                ),
            )
            times_mcgpu.append(time.time() - start_time_mcgpu)

            # Reconstruction with parallelproj
            start_time_decay_reconstruction = time.time()
            reconstructed_images = dynamic_decay_reconstruction_no_fit(
                merged_raw_file,
                img_shape=cropped_shape,
                voxel_size=voxel_size,
                scanner=scanner,
                num_subsets=num_subsets,
                osem_iterations=osem_iterations,
                body_mask=body_mask,
                sensitivity_array=sensitivity_array,
                end_time=FINAL_TIME - INITIAL_TIME,
                frame_duration=frame_duration,
                tumor_xmin=tumor_xmin,
                tumor_xmax=tumor_xmax,
                tumor_ymin=tumor_ymin,
                tumor_ymax=tumor_ymax,
                tumor_zmin=tumor_zmin,
                tumor_zmax=tumor_zmax,
                tumor_mask_dilated=tumor_mask_dilated,
                reduction_factor=REDUCTION_FACTOR,
                patient_info_path=patient_info_path,
            )  # voxel_size in cm

            os.remove(merged_raw_file)

            # Saving activity and regions
            np.save(
                os.path.join(scanner_folder, f"noisy_decay/plan{plan_num}.npy"),
                reconstructed_images,
            )

            times_decay_reconstruction.append(
                time.time() - start_time_decay_reconstruction
            )
            # plot the central slice of the three saved arrays in three imshow rows
            CT_tumor_cropped = np.load(CT_npy_path)
            lower_bound = 1 / maximum_mean_lives[PET_isotope]
            upper_bound = 1 / minimum_mean_lives[PET_isotope]
            lower_bound_frames = 0.0
            upper_bound_frames = reconstructed_images.max()
            if plan_num < 30:
                regions_plot_file = os.path.join(
                    scanner_folder, f"images/plan{plan_num}_regions.jpg"
                )
                # Plot slice
                NUM_PLOTS = 6
                fig, ax = plt.subplots(1, NUM_PLOTS, figsize=(20, 7))
                ax[0].imshow(
                    CT_tumor_cropped[:, CT_tumor_cropped.shape[1] // 2, :].T,
                    cmap="gray",
                    vmin=-125,
                    vmax=225,
                )
                ax[0].set_title("CT")
                ax[0].axis("off")
                ax[1].imshow(
                    reconstructed_images[
                        0, :, clean_decay_map.shape[1] // 2, :
                    ].T.get(),
                    cmap="inferno",
                    vmin=lower_bound_frames,
                    vmax=upper_bound_frames,
                )
                ax[1].set_title("Frame 1")
                ax[1].axis("off")
                ax[2].imshow(
                    reconstructed_images[
                        2, :, clean_decay_map.shape[1] // 2, :
                    ].T.get(),
                    cmap="inferno",
                    vmin=lower_bound_frames,
                    vmax=upper_bound_frames,
                )
                ax[2].set_title("Frame 3")
                ax[2].axis("off")
                ax[3].imshow(
                    reconstructed_images[
                        4, :, clean_decay_map.shape[1] // 2, :
                    ].T.get(),
                    cmap="inferno",
                    vmin=lower_bound_frames,
                    vmax=upper_bound_frames,
                )
                ax[3].set_title("Frame 5")
                ax[3].axis("off")

                decay_map_plot = ax[4].imshow(
                    clean_decay_map[:, clean_decay_map.shape[1] // 2, :].T.get(),
                    cmap="inferno",
                    vmin=lower_bound,
                    vmax=upper_bound,
                )
                ax[4].set_title("Clean decay")
                ax[4].axis("off")

                region_type_plot = ax[5].imshow(
                    region_types_map[:, region_types_map.shape[1] // 2, :].T.get(),
                    cmap=cm.get_cmap("tab20", NUM_REGION_TYPES),
                    vmin=0,
                    vmax=NUM_REGION_TYPES,
                )
                ax[5].set_title("Region types")
                ax[5].axis("off")

                for num_plot in range(NUM_PLOTS):
                    ax[num_plot].imshow(
                        outline_masked, alpha=1.0, vmin=0, vmax=1.0, cmap="winter"
                    )

                # add colorbar for every plot, make height smaller
                for num_plot in range(NUM_PLOTS):
                    fig.colorbar(
                        ax[num_plot].images[0],
                        ax=ax[num_plot],
                        orientation="horizontal",
                        fraction=0.046,
                    )

                plt.tight_layout()
                plt.savefig(regions_plot_file)

            times_iteration.append(time.time() - start_times_iteration)

            # save timings and means and stds
            with open(os.path.join(timings_dir, f"timings_{FILE_ID}.json"), "w") as f:
                json.dump(
                    {
                        "times_mcgpu": times_mcgpu,
                        "times_region_generation": times_region_generation,
                        "times_decay_reconstruction": times_decay_reconstruction,
                        "times_iteration": times_iteration,
                    },
                    f,
                )

        regions = regions[
            :-1
        ]  # Remove the last region, which is the outside of the tumor
        for idx, region in enumerate(regions):
            regions[idx] = region[
                tumor_xmin:tumor_xmax, tumor_ymin:tumor_ymax, tumor_zmin:tumor_zmax
            ]  # Crop to the tumor region
        np.save(os.path.join(dataset_folder, f"regions/plan{plan_num}.npy"), regions)
        del reconstructed_images, clean_decay_map
        shutil.rmtree(plan_i_location)

    gc.collect()
