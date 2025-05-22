""" This script trains a deep learning model to estimate the corrected washout parameter maps
from uncorrected parameter maps.
RUN WITH CONDA ENVIRONMENT prototwin-pet-washout (environment.yml,
install for any PC with conda env create -f environment.yml)
"""

import os
import sys
import torch
import numpy as np
from utils_train import train
from utils_test import test
from utils_model import (
    set_seed,
    CustomMinMaxScaler,
    Reshape3D,
    DecayMapDataset,
    plot_sample,
    custom_collate_fn,
    dataset_statistics,
    save_model_complexity
)
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotated,
    Rand3DElasticd,
)
from models.nnFormer.nnFormer_DPB import nnFormer
from torch.utils.data import DataLoader, Subset

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# ----------------------------------------------------------------------------------------------------------------------------------------
# USER-DEFINED PR0TOTWIN-PET-WASHOUT PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------------------------

seed = 42  # Set the seed for reproducibility
scanner = "vision"  # 'vision' or 'quadra' scanner

train_patient_names = [
    "HN-CHUM-013",  # T1 female
    "HN-CHUM-015",  # T1 male
    "HN-CHUM-017",  # T3 male,
    "HN-CHUM-018",  # T2 female
    "HN-CHUM-022",  # T2 male
    "HN-CHUM-053",  # T4 male
    "HN-CHUM-055",  # T3 female
    "HN-CHUM-060",  # T4 female
]

val_patient_names = ["HN-CHUM-010", "HN-CHUM-007"]
test_patient_names = [
    "HN-CHUM-010",  # T1 male
    "HN-CHUM-007",  # T2 female
    "HN-CHUM-027",  # T3 female
    "HN-CHUM-021",  # T4 male
]
# test_patient_names = ['LIVER']  # CORT dataset

dataset_num = 2
model_name = f"dataset{dataset_num}-nnFormer-v2"  # DEFINE THE MODEL NAME

voxel_size = (1.9531, 1.9531, 1.5)  # (mm) Image resolution, for head
train_fraction = 0.75  # Fraction of the dataset used for training
val_fraction = (
    0.1  # Fraction of the dataset used for validation (the rest is used for testing)
)
learning_rate = 1e-4  # Learning rate for the optimizer
# For epistemic uncertainty estimation, the model is trained with MC dropout method
epistemic_uncertainty = True
drop_rate = 0.2
attn_drop_rate = 0.1
mc_dropout_iters = 20

# For aleatoric uncertainty estimation, the model is trained with the loss function that includes the uncertainty term
aleatoric_uncertainty = True

train_model_flag = (
    True  # Set True to train  model, False to only test an already trained model
)

# -----------------------------------------------------------------------------------------------------------------------------------------

image_type = "decay"

set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, " : ", torch.cuda.get_device_name(torch.cuda.current_device()))

# Creating the images folder
images_dir = os.path.join(script_dir, f"../images/{model_name}")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Creating the dataset
base_dir = os.path.join(script_dir, "../data")

input_dir_name = f"{scanner}/noisy_{image_type}"
regions_dir_name = "regions"
sample_img = np.load(
    os.path.join(
        base_dir,
        test_patient_names[0],
        f"dataset{dataset_num}",
        input_dir_name,
        "plan0.npy",
    )
)
img_size = tuple(sample_img.shape[1:])
in_channels = sample_img.shape[0]

num_samples = 75  # int(len(os.listdir(os.path.join(base_dir, test_patient_names[0], f"dataset{dataset_num}", input_dir_name))))
print("Number of samples per patient: ", num_samples)
scaling = "min-max"
min_mean_life = 750.0  # highly vascularized tumor regions
max_mean_life = (
    1765.0  # {'C11': 1765.0}  # only physical washout, no biological washout
)

min_decay = 1 / max_mean_life  # Working with the decay constant
max_decay = 1 / min_mean_life

# CT
CT_scaling_transform = CustomMinMaxScaler(min_val=-1000, max_val=3000)
CT_flag = False  # To train with CT set to true
if CT_flag:
    in_channels += 1
    keys = ["input", "output", "CT"]  # augmentations applied to the CT image as well
else:
    keys = ["input", "output"]

# Create dataset applying the transforms
reshape_size = (64, 64, 64)
# # Downsample each dimension to closest multiple of 32
# reshape_size = [32 * (dim // 32) for dim in img_size]
print("Reshape size: ", reshape_size)

scaling_factors = np.array(img_size) / np.array(reshape_size)
reshape_voxel_size = voxel_size * scaling_factors

# Transformations
scaling_transform = CustomMinMaxScaler(min_val=min_decay, max_val=max_decay)
reshape_transform = Reshape3D(size=reshape_size, original_size=img_size)

scaling_input_dir_name = f"dataset{dataset_num}/{scanner}/noisy_{image_type}"
global_max, global_min = dataset_statistics(
    base_dir,
    train_patient_names,
    scaling_input_dir_name,
    scaling,
    num_samples=num_samples,
)
input_scaling_transform = CustomMinMaxScaler(min_val=global_min, max_val=global_max)

# Augmentations
rotation_angle = 10 * np.pi / 180  # converted to radians
train_augmentations = Compose(
    [
        # RandSpatialCropd(keys=keys, roi_size=reshape_size, random_center=True, random_size=False),  # Let's try cropping for training and then reshaping for testing, might not work because test images will have smaller pixels
        RandFlipd(keys=keys, spatial_axis=0, prob=0.3),
        RandFlipd(keys=keys, spatial_axis=1, prob=0.3),
        RandFlipd(keys=keys, spatial_axis=2, prob=0.3),
        RandRotated(
            keys=keys,
            range_x=rotation_angle,
            range_y=rotation_angle,
            range_z=rotation_angle,
            prob=0.5,
        ),
        Rand3DElasticd(
            keys=keys, prob=0.2, sigma_range=(6, 10), magnitude_range=(100, 150)
        ),
    ]
)

output_dir_name = f"clean_{image_type}"
num_classes = 1
results_dir = os.path.join(
    script_dir, f"models/test-results/{model_name}-results-decay-rates.txt"
)

print(
    "Number of pairs for training : ",
    num_samples * len(train_patient_names),
    ", validation: ",
    num_samples * len(val_patient_names),
    " , and testing: ",
    num_samples * len(test_patient_names),
)

train_dataset = DecayMapDataset(
    base_dir=base_dir,
    input_dir_name=input_dir_name,
    output_dir_name=output_dir_name,
    patient_names=train_patient_names,
    dataset_num=dataset_num,
    num_samples=num_samples,
    scaling_transform=scaling_transform,
    input_scaling_transform=input_scaling_transform,
    input_CT=False,
    # reshape_transform=reshape_transform,
    CT_scaling_transform=CT_scaling_transform,
    augmentations=train_augmentations,
    num_classes=num_classes,
)

val_dataset = DecayMapDataset(
    base_dir=base_dir,
    input_dir_name=input_dir_name,
    output_dir_name=output_dir_name,
    patient_names=val_patient_names,
    dataset_num=dataset_num,
    num_samples=num_samples,
    scaling_transform=scaling_transform,
    input_scaling_transform=input_scaling_transform,
    input_CT=False,
    CT_scaling_transform=CT_scaling_transform,
    # reshape_transform=reshape_transform,
    num_classes=num_classes,
)

test_dataset = DecayMapDataset(
    base_dir=base_dir,
    input_dir_name=input_dir_name,
    output_dir_name=output_dir_name,
    patient_names=test_patient_names,
    dataset_num=dataset_num,
    num_samples=num_samples,
    scaling_transform=scaling_transform,
    input_scaling_transform=input_scaling_transform,
    input_CT=True,
    CT_scaling_transform=CT_scaling_transform,
    # reshape_transform=reshape_transform,
    regions_dir_name=regions_dir_name,  # provided to test metrics for each region
    num_classes=num_classes,
)


# Create DataLoaders for training
batch_size = 8
num_workers = 1
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=custom_collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=custom_collate_fn,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=custom_collate_fn,
)

# Create the model
model = nnFormer(
    crop_size=reshape_size,
    embedding_dim=128,  # 96,
    input_channels=in_channels,
    num_classes=num_classes,
    depths=[2, 2, 2, 2],  # [2, 2, 2, 2],
    num_heads=[8, 16, 32, 64],  # [6, 12, 24, 48],
    drop_rate=drop_rate,
    attn_drop_rate=attn_drop_rate,
    aleatoric_uncertainty=aleatoric_uncertainty,
).to(device)

model_dir = os.path.join(script_dir, f"models/trained-models/{model_name}.pth")

timing_dir = os.path.join(
    script_dir, f"models/training-times/training-time-{model_name}.txt"
)
losses_dir = os.path.join(script_dir, f"models/losses/{model_name}-loss.csv")
n_epochs = 1500
patience = 150

accumulation_steps = max(
    4 // batch_size, 1
)  # number of batches before taking an optimizer step, trying to stabilize trainng for batch_size=1 used for the head dataset

if train_model_flag:
    trained_model = train(
        model,
        train_loader,
        val_loader,
        epochs=n_epochs,
        patience=patience,
        learning_rate=learning_rate,
        model_dir=model_dir,
        timing_dir=timing_dir,
        save_plot_dir=images_dir,
        losses_dir=losses_dir,
        accumulation_steps=accumulation_steps,
        aleatoric_uncertainty=aleatoric_uncertainty,
    )
else:
    # Loading the trained model
    # model_dir = os.path.join(script_dir, f"models/trained-models/dataset{dataset_num}-nnFormer-v1.pth")  # To test models on specific data not used for training, uncomment
    trained_model = torch.load(model_dir, map_location=torch.device(device))
    if hasattr(trained_model, "module"):
        trained_model = (
            trained_model.module
        )  # For models trained with DataParallel (multiple GPUs)
    # trained_model.aleatoric_uncertainty = aleatoric_uncertainty  # Had to force it for some reason for a model trained without the uncertainty

# Snippet to reorder the dataset so that a sample from each patient is tested in sequence
reordered_indices = []
for i in range(num_samples):
    for j in range(len(test_patient_names)):
        reordered_indices.append(i + j * num_samples)
reordered_indices[0] = reordered_indices[3]
reordered_indices[2] = reordered_indices[4]

test_dataset = Subset(test_dataset, reordered_indices)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=custom_collate_fn,
)

# Plotting slices of the dose
num_plots = 3
plane = "y"
num_slice = reshape_size[1] // 2
plot_sample(
    trained_model,
    test_loader,
    device,
    save_plot_dir=images_dir,
    transforms=scaling_transform,
    input_transforms=input_scaling_transform,
    CT_transforms=CT_scaling_transform,
    num_slice=num_slice,
    num_plots=num_plots,
    min_decay=min_decay,
    max_decay=max_decay,
    num_classes=num_classes,
    epistemic_uncertainty=epistemic_uncertainty,
    mc_dropout_iters=mc_dropout_iters,
    aleatoric_uncertainty=aleatoric_uncertainty,
    no_fit=True,  # the model is trained using the five PET frames rather than the noisy washout map
    plot_frames=True,
)

# Testing the model
test(
    trained_model,
    test_loader,
    device,
    results_dir=results_dir,
    transforms=scaling_transform,
    save_plot_dir=images_dir,
    voxel_size=reshape_voxel_size,
    min_decay=min_decay,
    max_decay=max_decay,
    num_classes=num_classes,
    epistemic_uncertainty=epistemic_uncertainty,
    mc_dropout_iters=mc_dropout_iters,
    aleatoric_uncertainty=aleatoric_uncertainty,
    no_fit=True,  # the model is trained using the five PET frames rather than the noisy washout map
)

# Save the model complexity, comment out if running out of GPU memory
model_sizes_txt = os.path.join(
    script_dir, f"models/model-sizes/{model_name}-model-sizes.txt"
)
img_size_tensor = (in_channels,) + img_size
save_model_complexity(trained_model, img_size=img_size_tensor, model_name=model_name, model_sizes_txt=model_sizes_txt)
