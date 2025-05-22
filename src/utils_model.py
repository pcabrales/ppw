""" This file contains utility functions for the model training and testing.
"""

import os
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import array_api_compat.cupy as xp
from cupyx.scipy.ndimage import binary_dilation
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm, font_manager
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, ListedColormap

# import matplotlib.ticker as ticker
from ptflops import get_model_complexity_info

script_dir = os.path.dirname(os.path.abspath(__file__))
# font_path = os.path.join(script_dir, '../images/Times_New_Roman.ttf')
font_path = os.path.join(script_dir, "../images/Helvetica.ttf")
font_manager.fontManager.addfont(font_path)


def set_seed(seed):
    """
    Set all the random seeds to a fixed value to take out any randomness
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    xp.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    return True


def enable_dropout(model):
    """Function to enable the dropout layers during test-time."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


class CustomMinMaxScaler:
    """
    Custom Min-Max Scaler, with specific min_val and max_val and inverse function
    """

    def __init__(self, min_val, max_val):
        """ Initialize the scaler with min and max values"""
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        ''' Apply the min-max scaling to the input tensor'''
        return (tensor - self.min_val) / (self.max_val - self.min_val)

    def inverse(self, tensor):
        ''' Inverse function to get the original values back'''
        return tensor * (self.max_val - self.min_val) + self.min_val

    def inverse_variance(self, tensor):
        """For variance we are only affecting the scale, not the offset"""
        return tensor * (self.max_val - self.min_val) ** 2


class Reshape3D(object):
    """
    Reshape the input tensor to 3D
    """

    def __init__(self, size, original_size):
        ''' Initialize the Reshape3D class with the target size and original size'''
        self.size = size  # Tuple like (64, 64, 64)
        self.original_size = original_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, tensor):
        ''' Apply the reshape transform to the input tensor'''
        data_resized = F.interpolate(
            tensor, size=self.size, mode="trilinear", align_corners=False
        )
        return data_resized

    def inverse(self, tensor):
        ''' Inverse function to get the original size back'''
        data_resized = F.interpolate(
            tensor, size=self.original_size, mode="trilinear", align_corners=False
        )
        return data_resized


def custom_collate_fn(batch):
    """Custom collate function to stack the input and output maps into tensor batches"""
    input_maps, output_maps, regions_map_tensor_lists, tumor_mask = zip(*batch)
    # Stack input_maps and output_maps as they are tensors of the same size
    input_maps = torch.stack(input_maps, dim=0)
    output_maps = torch.stack(output_maps, dim=0)
    tumor_mask = torch.stack(tumor_mask, dim=0)
    # regions_map_tensor_lists is a tuple of lists (variable lengths), so we keep them as is
    return input_maps, output_maps, regions_map_tensor_lists, tumor_mask


class DecayMapDataset(Dataset):
    """
    Dataset class for the decay maps
    """

    def __init__(
        self,
        base_dir,
        input_dir_name,
        output_dir_name,
        patient_names,  # list of patient names
        dataset_num,  # dataset number
        num_samples,  # number of samples to load per patient
        scaling_transform=None,
        input_scaling_transform=None,
        channel_1_scaling_transform=None,
        input_CT=True,  # Flag to include the CT as an input channel
        CT_scaling_transform=None,
        reshape_transform=None,
        augmentations=None,
        regions_dir_name="regions",
        num_classes=None,
    ):

        self.base_dir = base_dir
        self.input_dir_name = input_dir_name
        self.output_dir_name = output_dir_name
        self.patient_names = patient_names
        self.dataset_num = dataset_num
        self.num_samples = num_samples
        self.scaling_transform = scaling_transform
        self.input_CT = input_CT
        if input_scaling_transform is None:
            self.input_scaling_transform = scaling_transform
        else:
            self.input_scaling_transform = input_scaling_transform
        self.channel_1_scaling_transform = channel_1_scaling_transform
        self.CT_scaling_transform = CT_scaling_transform
        self.reshape_transform = reshape_transform
        self.augmentations = augmentations
        self.regions_dir_name = regions_dir_name
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples * len(self.patient_names)

    def __getitem__(self, idx):
        patient_name = self.patient_names[idx // self.num_samples]
        idx_patient = idx % self.num_samples
        input_map_dir = os.path.join(
            self.base_dir,
            patient_name,
            f"dataset{self.dataset_num}",
            self.input_dir_name,
            f"plan{idx_patient}.npy",
        )
        input_map = torch.tensor(np.load(input_map_dir), dtype=torch.float32)
        if (
            input_map.dim() == 3
        ):  # if inputs are all PET frames instead of Uncorrected washout map, don't unsqueeze
            input_map = input_map.unsqueeze(0)
        output_map_dir = os.path.join(
            self.base_dir,
            patient_name,
            f"dataset{self.dataset_num}",
            self.output_dir_name,
            f"plan{idx_patient}.npy",
        )
        output_map = torch.tensor(
            np.load(output_map_dir), dtype=torch.float32
        ).unsqueeze(0)
        CT_dir = os.path.join(
            self.base_dir, patient_name, f"dataset{self.dataset_num}", "CT.npy"
        )
        if not os.path.exists(CT_dir):
            CT_dir = os.path.join(self.base_dir, patient_name, "CT.npy")
        CT = torch.tensor(np.load(CT_dir), dtype=torch.float32).unsqueeze(0)
        tumor_mask_dir = os.path.join(
            self.base_dir, patient_name, f"dataset{self.dataset_num}", "tumor_mask.npy"
        )
        tumor_mask = torch.tensor(
            np.load(tumor_mask_dir), dtype=torch.float32
        ).unsqueeze(0)

        if self.augmentations is not None:
            data = {"input": input_map, "output": output_map, "CT": CT}
            data = self.augmentations(data)  # APPLY AUGMENTATIONS
            input_map = data["input"]
            output_map = data["output"]
            CT = data["CT"]

        if self.reshape_transform:
            input_map = self.reshape_transform(input_map.unsqueeze(0)).squeeze(0)
            output_map = self.reshape_transform(output_map.unsqueeze(0)).squeeze(0)
            CT = self.reshape_transform(CT.unsqueeze(0)).squeeze(0)
            tumor_mask = self.reshape_transform(tumor_mask.unsqueeze(0)).squeeze(
                0
            )  # tumor mask is not used in the training (no augmentations applied), but we need to reshape it for the plotting and testing
            tumor_mask = tumor_mask > 0.5  # converting to boolean
        tumor_mask = tumor_mask.to(torch.int)

        if self.input_scaling_transform is not None:
            if (
                self.channel_1_scaling_transform is not None
            ):  # applying different scaling to the second channel (which can be the initial activity, for example, as opposed to the washout rate)
                input_map[1] = self.channel_1_scaling_transform(
                    input_map[1].unsqueeze(0)
                ).squeeze(0)
                input_map[0] = self.input_scaling_transform(
                    input_map[0].unsqueeze(0)
                ).squeeze(0)
            else:
                input_map = self.input_scaling_transform(
                    input_map.unsqueeze(0)
                ).squeeze(0)

        if self.scaling_transform is not None and self.num_classes == 1:
            output_map = self.scaling_transform(output_map.unsqueeze(0)).squeeze(0)

        if self.CT_scaling_transform is not None:
            CT = self.CT_scaling_transform(CT.unsqueeze(0)).squeeze(0)

        # Currently, no augmentations are applied to regions because they are only used for testing
        if self.regions_dir_name is not None:
            regions_map_dir = os.path.join(
                self.base_dir,
                patient_name,
                f"dataset{self.dataset_num}",
                self.regions_dir_name,
                f"plan{idx_patient}.npy",
            )
            regions_map = np.load(os.path.join(regions_map_dir))
            regions_map_tensor_list = []
            if self.reshape_transform is None:
                self.reshape_transform = (
                    lambda x: x
                )  # if no reshape transform is given, we assume the data is already reshaped and apply the identity transform
            for region in regions_map:
                regions_map_tensor_list.append(
                    self.reshape_transform(
                        torch.tensor(region, dtype=torch.float32)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    .squeeze(0)
                    .squeeze(0)
                    > 0.5  # interpolating the region mask as float and converting to boolean
                )
        else:  # if no regions are given, make it into something that can be zipped
            regions_map_tensor_list = [torch.tensor([0])]

        if self.input_CT:
            input_map = torch.cat((input_map, CT), dim=0)
        return input_map, output_map, regions_map_tensor_list, tumor_mask


def dataset_statistics(
    base_dir, patient_names, dir_name, scaling="min-max", num_samples=5, channel=None
):
    """Function to get dataset-wide statistics for scaling"""
    if scaling == "min-max" or scaling == "minmax":
        # Initialize min and max with extreme values
        global_min = np.inf
        global_max = -np.inf

        for patient_name in patient_names:
            dataset_dir = os.path.join(base_dir, patient_name, dir_name)
            for file_num in range(num_samples):
                file_path = os.path.join(dataset_dir, f"plan{file_num}.npy")
                data = np.load(file_path, mmap_mode="r")  # Memory-efficient loading
                if channel is not None:
                    data = data[
                        channel
                    ]  # it may be that the input data has multiple channels, such as one for washout rate and other for initial activity, and they have different statistics
                # Update global min and max
                global_min = min(global_min, np.min(data))
                global_max = max(global_max, np.max(data))

            print(f"Max. pixel value: {global_max:0.11f}")
            print(f"Min. pixel value: {global_min:0.11f}")
            return [global_max, global_min]
    else:
        raise ValueError(
            "Scaling not recognized. Please choose between standard, minmax or robust."
        )


def plot_sample(
    trained_model,
    test_loader,
    device,
    save_plot_dir,
    transforms=None,
    input_transforms=None,
    CT_transforms=None,
    plane="coronal",
    num_slice=32,
    num_plots=1,
    min_decay=0,
    max_decay=1,
    region_types_flag=False,
    num_classes=1,
    edges_region_types=None,  # to convert output decay map to region types
    epistemic_uncertainty=False,
    mc_dropout_iters=10,
    aleatoric_uncertainty=False,
    estimating_washout_fraction=False,
    no_fit=False,  # if True, the model is trained using the five PET frames rather than the Uncorrected washout map
    plot_frames=False,  # if no_fit is True, this flag will plot the PET frames as separate figures
):
    """Plot the maps, uncertainties and errors for a given number of samples from the test set"""

    if input_transforms is None:
        input_transforms = transforms

    # Set the colormap for the region types
    cmap_region_types = ListedColormap(
        ["#377eb8", "#984ea3", "#ff7f00", '#ffff33', '#a65628'])

    if estimating_washout_fraction:
        max_decay *= 100  # converting to percentage
    else:
        max_decay = (
            max_decay - min_decay
        )  # only leaving  the biological decay rates, removing physical decay
        max_decay *= 60  # converting to 1/minutes

    trained_model.eval()
    if epistemic_uncertainty:
        enable_dropout(trained_model)
    (
        input,
        output,
        target,
        CT,
        tumor_mask,
        error_input,
        error_output,
        epistemic_uncertainty_array,
        aleatoric_uncertainty_array,
    ) = ([], [], [], [], [], [], [], [], [])

    with torch.no_grad():
        for i, (batch_input, batch_target, _, batch_tumor_mask) in enumerate(
            test_loader
        ):
            batch_input = batch_input.to(device)
            batch_CT = batch_input[:, -1, :, :, :].unsqueeze(1)
            batch_input = batch_input[:, :-1, :, :, :]
            batch_target = batch_target.to(device)
            batch_tumor_mask = batch_tumor_mask.to(device)

            if CT_transforms is not None:
                batch_CT = CT_transforms.inverse(batch_CT)

            if epistemic_uncertainty:
                batch_output = torch.zeros((mc_dropout_iters, *batch_target.shape)).to(
                    device
                )
                if aleatoric_uncertainty:
                    batch_aleatoric_variance = torch.zeros(
                        (mc_dropout_iters, *batch_target.shape)
                    ).to(device)
                for mc_dropout_iter in tqdm(
                    range(mc_dropout_iters), desc="MC Dropout Iterations"
                ):
                    set_seed(mc_dropout_iter)
                    if aleatoric_uncertainty:
                        batch_output[mc_dropout_iter] = trained_model(batch_input)[
                            :, 0
                        ].unsqueeze(1)
                        batch_aleatoric_variance[mc_dropout_iter] = trained_model(
                            batch_input
                        )[:, 1].unsqueeze(1)
                    else:
                        batch_output[mc_dropout_iter] = trained_model(batch_input)
                    if transforms:
                        batch_output[mc_dropout_iter] = transforms.inverse(
                            batch_output[mc_dropout_iter]
                        )
                        if aleatoric_uncertainty:
                            batch_aleatoric_variance[mc_dropout_iter] = (
                                transforms.inverse_variance(
                                    batch_aleatoric_variance[mc_dropout_iter]
                                )
                            )
                batch_epistemic_uncertainty = batch_output.std(
                    dim=0
                )  # * 2  # 2 standard deviations
                batch_output = batch_output.mean(dim=0)
                if aleatoric_uncertainty:
                    batch_aleatoric_uncertainty = torch.sqrt(
                        batch_aleatoric_variance.mean(dim=0)
                    )
            else:
                if aleatoric_uncertainty:
                    batch_output = trained_model(batch_input)[:, 0].unsqueeze(1)
                    batch_aleatoric_variance = trained_model(batch_input)[
                        :, 1
                    ].unsqueeze(1)
                    if transforms:
                        batch_output = transforms.inverse(batch_output)
                        batch_aleatoric_variance = transforms.inverse_variance(
                            batch_aleatoric_variance
                        )
                    batch_aleatoric_uncertainty = torch.sqrt(batch_aleatoric_variance)
                else:
                    batch_output = trained_model(batch_input)[:, 0].unsqueeze(1)
                    if transforms:
                        batch_output = transforms.inverse(batch_output)

            if no_fit and plot_frames:
                plot_frames_dir = os.path.join(save_plot_dir, "PET_frames")
                if not os.path.exists(plot_frames_dir):
                    os.makedirs(plot_frames_dir)

                max_val = torch.max(batch_input[i])
                vmin_frame = 0
                vmax_frame = max_val
                for frame_num in range(batch_input.shape[1]):
                    frame = np.squeeze(batch_input[i, frame_num, :, :, :].cpu().numpy())
                    if plane == "coronal" or plane == "y":
                        frame_slice = np.flip(frame[:, num_slice, :].T, axis=0)
                    elif plane == "axial" or plane == "z":
                        frame_slice = frame[:, :, num_slice]
                    elif plane == "sagital" or plane == "x":
                        frame_slice = np.flip(frame[num_slice, :, :], axis=1).T
                    else:
                        raise ValueError("Plane must be coronal, sagittal or axial")
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    ax.axis("off")
                    ax.imshow(frame_slice, cmap="jet", vmin=vmin_frame, vmax=vmax_frame)
                    fig.savefig(
                        os.path.join(plot_frames_dir, f"frame{frame_num}.png"),
                        dpi=600,
                        bbox_inches="tight",
                        transparent=True,
                    )
                    plt.close(fig)

            if transforms:
                batch_input = input_transforms.inverse(
                    batch_input[:, 0, :, :, :].unsqueeze(1)
                )
                if estimating_washout_fraction:
                    batch_input *= 100  # converting to percentage
                else:
                    batch_input -= min_decay  # removing the physical decay
                    batch_input *= 60  # converting to 1/minutes
                if region_types_flag:
                    if (
                        edges_region_types is None
                    ):  # if bucketized, the tensor already has one channel with an integer values for each region type
                        _, batch_output = torch.max(batch_output, dim=1)
                        batch_output = batch_output.unsqueeze(1)
                    else:
                        batch_output = (
                            torch.bucketize(batch_output, edges_region_types) - 1
                        )
                        batch_output[batch_output == -1] = 0
                        batch_output[batch_output == num_classes] = num_classes - 1
                        batch_output = (
                            num_classes - 1 - batch_output
                        )  # to have the same order as the original region types (which are ordered in ascending order of mean life and in descending order of decay rate)

                        batch_target = (
                            torch.bucketize(batch_target, edges_region_types) - 1
                        )
                        batch_target[batch_target == -1] = 0
                        batch_target[batch_target == num_classes] = num_classes - 1
                        batch_target = (
                            num_classes - 1 - batch_target
                        )  # to have the same order as the original region types (which are ordered in ascending order of mean life and in descending order of decay rate)

                    error_output.append(
                        (batch_output - batch_target) == 0
                    )  # 1 if the output is correct, 0 otherwise
                else:
                    batch_target = transforms.inverse(batch_target)
                    if estimating_washout_fraction:
                        batch_output *= 100
                        batch_target *= 100
                        if epistemic_uncertainty:
                            batch_epistemic_uncertainty *= 100
                        if aleatoric_uncertainty:
                            batch_aleatoric_uncertainty *= 100
                    else:
                        batch_output -= min_decay
                        batch_target -= min_decay
                        batch_output *= 60
                        batch_target *= 60
                        if epistemic_uncertainty:
                            batch_epistemic_uncertainty *= 60
                        if aleatoric_uncertainty:
                            batch_aleatoric_uncertainty *= 60
                    error_input.append(torch.abs(batch_input - batch_target))
                    error_output.append(torch.abs(batch_output - batch_target))
            input.append(batch_input)
            output.append(batch_output)
            target.append(batch_target)
            CT.append(batch_CT)
            tumor_mask.append(batch_tumor_mask)
            if epistemic_uncertainty:
                epistemic_uncertainty_array.append(batch_epistemic_uncertainty)
            if aleatoric_uncertainty:
                aleatoric_uncertainty_array.append(batch_aleatoric_uncertainty)
            if (i + 1) * test_loader.batch_size >= num_plots:
                break
    input = torch.cat(input, dim=0).cpu().numpy()
    output = torch.cat(output, dim=0)
    target = torch.cat(target, dim=0)
    CT = torch.cat(CT, dim=0).cpu().numpy()
    tumor_mask = torch.cat(tumor_mask, dim=0)
    error_output = torch.cat(error_output, dim=0).cpu().numpy()
    if not region_types_flag:
        error_input = torch.cat(error_input, dim=0).cpu().numpy()
    if epistemic_uncertainty and not aleatoric_uncertainty:
        uncertainty_array = torch.cat(epistemic_uncertainty_array, dim=0)
    elif aleatoric_uncertainty and not epistemic_uncertainty:
        uncertainty_array = torch.cat(aleatoric_uncertainty_array, dim=0)
    elif epistemic_uncertainty and aleatoric_uncertainty:
        uncertainty_array = torch.sqrt(
            torch.cat(epistemic_uncertainty_array, dim=0) ** 2
            + torch.cat(aleatoric_uncertainty_array, dim=0) ** 2
        )

    if epistemic_uncertainty or aleatoric_uncertainty:
        uncertainty_array = uncertainty_array.cpu().numpy()
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    tumor_mask = tumor_mask.cpu().numpy()

    vmin_CT = -125
    vmax_CT = 225

    if estimating_washout_fraction:  # get maximum and minimum washout_fraction
        vmin_main = 100.0
        vmax_main = 0.0
    if region_types_flag:
        vmax_error = 1
    else:
        vmax_error = 0.0
        # Loop to get maximum error or uncertainty for the colorbar
        for i in range(num_plots):
            if plane == "coronal" or plane == "y":
                tumor_mask_slice = tumor_mask[i, 0, :, num_slice, :]
                error_input_slice = (
                    error_input[i, 0, :, num_slice, :] * tumor_mask_slice
                )
                error_output_slice = (
                    error_output[i, 0, :, num_slice, :] * tumor_mask_slice
                )
                input_slice = input[i, 0, :, num_slice, :][
                    tumor_mask_slice.astype(bool)
                ]
                output_slice = output[i, 0, :, num_slice, :][
                    tumor_mask_slice.astype(bool)
                ]
                target_slice = target[i, 0, :, num_slice, :][
                    tumor_mask_slice.astype(bool)
                ]
                if epistemic_uncertainty or aleatoric_uncertainty:
                    uncertainty_slice = (
                        uncertainty_array[i, 0, :, num_slice, :] * tumor_mask_slice
                    )
                else:
                    uncertainty_slice = np.zeros_like(input_slice)
            elif plane == "axial" or plane == "z":
                tumor_mask_slice = tumor_mask[i, 0, :, :, num_slice]
                error_input_slice = (
                    error_input[i, 0, :, :, num_slice] * tumor_mask_slice
                )
                error_output_slice = (
                    error_output[i, 0, :, :, num_slice] * tumor_mask_slice
                )
                input_slice = input[i, 0, :, :, num_slice][
                    tumor_mask_slice.astype(bool)
                ]
                output_slice = output[i, 0, :, :, num_slice][
                    tumor_mask_slice.astype(bool)
                ]
                target_slice = target[i, 0, :, :, num_slice][
                    tumor_mask_slice.astype(bool)
                ]
                if epistemic_uncertainty or aleatoric_uncertainty:
                    uncertainty_slice = (
                        uncertainty_array[i, 0, :, :, num_slice] * tumor_mask_slice
                    )
                else:
                    uncertainty_slice = np.zeros_like(input_slice)
            elif plane == "sagital" or plane == "x":
                tumor_mask_slice = tumor_mask[i, 0, num_slice, :, :]
                error_input_slice = (
                    error_input[i, 0, num_slice, :, :] * tumor_mask_slice
                )
                error_output_slice = (
                    error_output[i, 0, num_slice, :, :] * tumor_mask_slice
                )
                input_slice = input[i, 0, num_slice, :, :][
                    tumor_mask_slice.astype(bool)
                ]
                output_slice = output[i, 0, num_slice, :, :][
                    tumor_mask_slice.astype(bool)
                ]
                target_slice = target[i, 0, num_slice, :, :][
                    tumor_mask_slice.astype(bool)
                ]
                if epistemic_uncertainty or aleatoric_uncertainty:
                    uncertainty_slice = (
                        uncertainty_array[i, 0, num_slice, :, :] * tumor_mask_slice
                    )
                else:
                    uncertainty_slice = np.zeros_like(input_slice)
            else:
                raise ValueError("Plane must be coronal, sagittal or axial")
            if no_fit:
                vmax_error = max(np.max(error_output_slice), vmax_error)
            else:
                vmax_error = max(
                    np.max(error_input_slice), np.max(error_output_slice), vmax_error
                )
            if epistemic_uncertainty or aleatoric_uncertainty:
                vmax_error = max(vmax_error, np.max(uncertainty_slice))
            if estimating_washout_fraction:
                vmax_main = max(
                    np.max(input_slice),
                    np.max(output_slice),
                    np.max(target_slice),
                    vmax_main,
                )
                vmin_main = min(
                    np.min(input_slice),
                    np.min(output_slice),
                    np.min(target_slice),
                    vmin_main,
                )
    if not estimating_washout_fraction:
        vmin_main = 0.0
        vmax_main = max_decay
    sns.set_style("white")
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams["font.sans-serif"] = "Helvetica"
    font_size = 25
    if region_types_flag:
        num_cols = 5
        fig, axs = plt.subplots(
            num_plots, num_cols, figsize=(num_cols * 2.4, num_plots * 4.2)
        )
        column_titles = [
            "CT",
            "Uncorrected",
            "Ground Truth\nRegion Types",
            "Predicted\nRegion Types",
            "Prediction\nError",
        ]
    else:
        num_cols = 6
        column_titles = [
            "CT",
            "Ground Truth (GT)",
            "Uncorrected",
            "Corrected",
            "Absolute Error\n|GT - Uncorrected|",
            "Absolute Error\n|GT - Corrected|",
        ]
        if epistemic_uncertainty or aleatoric_uncertainty:
            num_cols += 1  # 2
            column_titles.append("Uncertainty")
            # column_titles.append('Sparsification\nError')
        fig, axs = plt.subplots(
            num_plots, num_cols, figsize=(num_cols * 2.85, num_plots * 3.6)
        )

    if num_plots == 1:
        axs = [axs]

    for i in range(num_plots):
        if plane == "coronal" or plane == "y":
            CT_slice = np.flip(CT[i, 0, :, num_slice, :].T, axis=0)
            input_slice = np.flip(input[i, 0, :, num_slice, :].T, axis=0)
            output_slice = np.flip(output[i, 0, :, num_slice, :].T, axis=0)
            target_slice = np.flip(target[i, 0, :, num_slice, :].T, axis=0)
            error_output_slice = np.flip(error_output[i, 0, :, num_slice, :].T, axis=0)
            if not region_types_flag:
                error_input_slice = np.flip(
                    error_input[i, 0, :, num_slice, :].T, axis=0
                )
            if epistemic_uncertainty or aleatoric_uncertainty:
                uncertainty_slice = np.flip(
                    uncertainty_array[i, 0, :, num_slice, :].T, axis=0
                )
            tumor_mask_slice = np.flip(tumor_mask[i, 0, :, num_slice, :].T, axis=0)
        elif plane == "axial" or plane == "z":
            CT_slice = CT[i, 0, :, :, num_slice]
            input_slice = input[i, 0, :, :, num_slice]
            output_slice = output[i, 0, :, :, num_slice]
            target_slice = target[i, 0, :, :, num_slice]
            error_output_slice = error_output[i, 0, :, :, num_slice]
            if not region_types_flag:
                error_input_slice = error_input[i, 0, :, :, num_slice]
            if epistemic_uncertainty or aleatoric_uncertainty:
                uncertainty_slice = uncertainty_array[i, 0, :, :, num_slice]
            tumor_mask_slice = tumor_mask[i, 0, :, :, num_slice]
        elif plane == "sagital" or plane == "x":
            CT_slice = np.flip(CT[i, 0, num_slice, :, :], axis=1).T
            input_slice = np.flip(input[i, 0, num_slice, :, :], axis=1).T
            output_slice = np.flip(output[i, 0, num_slice, :, :], axis=1).T
            target_slice = np.flip(target[i, 0, num_slice, :, :], axis=1).T
            error_output_slice = np.flip(error_output[i, 0, num_slice, :, :], axis=1).T
            if not region_types_flag:
                error_input_slice = np.flip(
                    error_input[i, 0, num_slice, :, :], axis=1
                ).T
            if epistemic_uncertainty or aleatoric_uncertainty:
                uncertainty_slice = np.flip(
                    uncertainty_array[i, 0, num_slice, :, :], axis=1
                ).T
            tumor_mask_slice = np.flip(tumor_mask[i, 0, num_slice, :, :], axis=1).T
        else:
            raise ValueError("Plane must be coronal, sagittal or axial")

        # Since we are only interested in the error inside the tumor, we will mask the error maps with the tumor mask
        if not region_types_flag:
            error_input_slice = error_input_slice * tumor_mask_slice
            error_output_slice = error_output_slice * tumor_mask_slice
            if epistemic_uncertainty or aleatoric_uncertainty:
                uncertainty_slice = uncertainty_slice * tumor_mask_slice
        else:
            error_output_slice[tumor_mask_slice == 0] = (
                True  # only the tumor region is considered for the error
            )
            output_slice[tumor_mask_slice == 0] = target_slice[tumor_mask_slice == 0]

        axs[i][0].imshow(CT_slice, cmap="gray", vmin=vmin_CT, vmax=vmax_CT)
        axs[i][0].axis("off")
        cmap = "inferno"
        if region_types_flag:
            axs[i][1].imshow(input_slice, cmap=cmap, vmin=0, vmax=max_decay)
            axs[i][1].axis("off")
            axs[i][2].imshow(
                target_slice, cmap=cmap_region_types, vmin=0, vmax=num_classes - 1
            )
            axs[i][2].axis("off")
            axs[i][3].imshow(
                output_slice, cmap=cmap_region_types, vmin=0, vmax=num_classes - 1
            )
            axs[i][3].axis("off")
            cmap_error = mcolors.ListedColormap(
                ["red", "green"]
            )  # if gamma is below 1.0, it is green, otherwise red
            axs[i][4].imshow(
                error_output_slice, cmap=cmap_error, vmin=0, vmax=vmax_error
            )
            axs[i][4].axis("off")
        else:
            axs[i][1].imshow(target_slice, cmap=cmap, vmin=vmin_main, vmax=vmax_main)
            axs[i][1].axis("off")
            axs[i][2].imshow(input_slice, cmap=cmap, vmin=vmin_main, vmax=vmax_main)
            axs[i][2].axis("off")
            axs[i][3].imshow(output_slice, cmap=cmap, vmin=vmin_main, vmax=vmax_main)
            axs[i][3].axis("off")
            cmap_error = "Reds"
            if epistemic_uncertainty or aleatoric_uncertainty:
                axs[i][6].imshow(
                    uncertainty_slice, cmap=cmap_error, vmin=0, vmax=vmax_error
                )
                axs[i][6].axis("off")

            axs[i][4].imshow(
                error_input_slice, cmap=cmap_error, vmin=0, vmax=vmax_error
            )
            axs[i][4].axis("off")
            axs[i][5].imshow(
                error_output_slice, cmap=cmap_error, vmin=0, vmax=vmax_error
            )
            axs[i][5].axis("off")

        tumor_outline = (
            tumor_mask_slice
            ^ binary_dilation(
                xp.array(tumor_mask_slice), iterations=2, brute_force=True
            ).get()
        )
        outline_masked = np.ma.masked_where(tumor_outline == 0, tumor_outline)

        if estimating_washout_fraction:
            dilated_tumor = input_slice >= 1.0  # space inside the dilated tumor volume
        else:
            dilated_tumor = input_slice >= 0.0
        outer_tumor_outline = (
            dilated_tumor
            ^ binary_dilation(
                xp.array(dilated_tumor), iterations=2, brute_force=True
            ).get()
        )
        outer_outline_masked = np.ma.masked_where(
            outer_tumor_outline == 0, outer_tumor_outline
        )

        for j in range(
            num_cols
        ):  # range(num_cols - 1 if plotting sparsification plot) last column is the error plot
            axs[i][j].imshow(outline_masked, alpha=1.0, vmin=0, vmax=1.0, cmap="winter")
        axs[i][0].imshow(outer_outline_masked, alpha=1.0, vmin=0, vmax=1.0, cmap="cool")

    # add titles
    for ax, title in zip(axs[0], column_titles):
        ax.set_title(title, fontsize=font_size)

    # CT COLORBAR
    if region_types_flag:
        save_plot_file = os.path.join(save_plot_dir, "voxel-types-maps.jpg")
        cbar_ax = fig.add_axes([0.04, 0.05, 0.14, 0.03])
    elif estimating_washout_fraction:
        save_plot_file = os.path.join(save_plot_dir, "washout-fraction-maps.jpg")
        cbar_ax = fig.add_axes([0.03, 0.12, 0.11, 0.03])
    else:
        save_plot_file = os.path.join(save_plot_dir, "washout-rate-maps.jpg")
        cbar_ax = fig.add_axes([0.03, 0.12, 0.11, 0.03])
    cbar = fig.colorbar(
        cm.ScalarMappable(cmap="gray", norm=Normalize(vmin=vmin_CT, vmax=vmax_CT)),
        cax=cbar_ax,
        orientation="horizontal",
    )
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label("(HU)", fontsize=font_size, labelpad=10)

    # DECAY RATE COLORBAR
    if region_types_flag:
        cbar_ax = fig.add_axes([0.24, 0.1, 0.13, 0.03])
    else:
        cbar_ax = fig.add_axes([0.185, 0.12, 0.36, 0.03])
    cbar = fig.colorbar(
        cm.ScalarMappable(
            cmap="inferno", norm=Normalize(vmin=vmin_main, vmax=vmax_main)
        ),
        cax=cbar_ax,
        orientation="horizontal",
    )
    cbar.ax.tick_params(
        labelsize=font_size
    )  # Set the colorbar ticks to scientific notation
    cbar.ax.ticklabel_format(style="sci", axis="x")
    if estimating_washout_fraction:
        cbar.set_label("Washout Fraction (%)", fontsize=font_size, labelpad=10)
    else:
        cbar.set_label(
            "Washout Rate" + r" $\lambda_B$ (min$^{-1}$)",
            fontsize=font_size,
            labelpad=10,
        )
    # cbar.locator = ticker.MaxNLocator(nbins=5)  # Set the number of ticks
    # cbar.update_ticks()

    # ERROR DECAY RATE COLORBAR
    if not region_types_flag:
        cbar_ax = fig.add_axes([0.6, 0.12, 0.37, 0.03])
        cbar = fig.colorbar(
            cm.ScalarMappable(cmap=cmap_error, norm=Normalize(vmin=0, vmax=vmax_error)),
            cax=cbar_ax,
            orientation="horizontal",
        )
        cbar.ax.tick_params(labelsize=font_size)
        # cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
        # cbar.locator = ticker.MaxNLocator(nbins=3)
        if estimating_washout_fraction:
            cbar.set_label("Washout Fraction (%)", fontsize=font_size, labelpad=10)
        else:
            cbar.set_label(
                "Washout Rate " + r" $\lambda_B$ (min$^{-1}$)",
                fontsize=font_size,
                labelpad=10,
            )

    # REGION TYPE AND REGION TYPE ERROR COLORBARS
    if region_types_flag:
        cbar_ax = fig.add_axes([0.44, 0.1, 0.32, 0.03])
        norm = Normalize(vmin=0, vmax=num_classes - 1)
        cbar = fig.colorbar(
            cm.ScalarMappable(cmap=cmap_region_types, norm=norm),
            cax=cbar_ax,
            orientation="horizontal",
        )
        cbar.set_label(
            label=r"     0       1       2       3       4     ", size=font_size
        )
        cbar.ax.tick_params(
            labelsize=0, grid_alpha=0, length=0
        )  # Remove the tick marks

        cbar_ax = fig.add_axes([0.8, 0.1, 0.16, 0.03])
        norm = Normalize(vmin=0, vmax=1)
        cbar = fig.colorbar(
            cm.ScalarMappable(cmap=cmap_error, norm=norm),
            cax=cbar_ax,
            orientation="horizontal",
        )
        cbar.set_label(label=r"Incorrect Correct", size=font_size - 1)
        cbar.ax.tick_params(labelsize=0, grid_alpha=0, length=0)

    text_1 = "Sample A"
    text_2 = "Sample B"
    text_3 = "Sample C"
    fig.text(
        0.0, 0.8, text_1, va="center", rotation="vertical", fontsize=font_size
    )  # , fontstyle='italic')
    fig.text(
        0.0, 0.54, text_2, va="center", rotation="vertical", fontsize=font_size
    )  # , fontstyle='italic')
    fig.text(
        0.0, 0.28, text_3, va="center", rotation="vertical", fontsize=font_size
    )  # , fontstyle='italic')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14, left=0.022)

    # SAVE PLOT
    # fig.savefig(save_plot_file, dpi=600, bbox_inches='tight')
    # fig.savefig(save_plot_file[:-3] + "eps", format='eps', bbox_inches='tight')
    # save as pdf
    fig.savefig(save_plot_file[:-3] + "pdf", format="pdf", bbox_inches="tight")


def plot_tumors(
    trained_model,
    test_loader,
    device,
    save_plot_dir,
    transforms=None,
    input_transforms=None,
    CT_transforms=None,
    plane="coronal",
    num_slice=32,
    min_decay=0,
    max_decay=1,
    region_types_flag=False,
    num_classes=1,
    edges_region_types=None,  # to convert output decay map to region types
    epistemic_uncertainty=False,
    mc_dropout_iters=10,
    aleatoric_uncertainty=False,
    estimating_washout_fraction=False,
    no_fit=False,  # if True, the model is trained using the five PET frames rather than the Uncorrected washout map
):
    """Plot the maps, cropped to the tumor, for the workflow image for the paper"""

    if input_transforms is None:
        input_transforms = transforms

    # Set the colormap for the region types
    cmap_region_types = ListedColormap(
        ["#377eb8", "#984ea3", "#ff7f00", "#ffff33", "#a65628"]
    )

    if not estimating_washout_fraction:
        max_decay = (
            max_decay - min_decay
        )  # only leaving  the biological decay rates, removing physical decay
        max_decay *= 60  # converting to 1/minutes

    trained_model.eval()
    if epistemic_uncertainty:
        enable_dropout(trained_model)
    with torch.no_grad():
        for _, (batch_input, batch_target, _, batch_tumor_mask) in enumerate(
            test_loader
        ):
            batch_input = batch_input.to(device)
            batch_CT = batch_input[:, -1, :, :, :].unsqueeze(1)
            batch_input = batch_input[:, :-1, :, :, :]
            batch_target = batch_target.to(device)
            batch_tumor_mask = batch_tumor_mask.to(device)

            if CT_transforms is not None:
                batch_CT = CT_transforms.inverse(batch_CT)

            if epistemic_uncertainty:
                batch_output = torch.zeros((mc_dropout_iters, *batch_target.shape)).to(
                    device
                )
                if aleatoric_uncertainty:
                    batch_aleatoric_variance = torch.zeros(
                        (mc_dropout_iters, *batch_target.shape)
                    ).to(device)
                for mc_dropout_iter in tqdm(
                    range(mc_dropout_iters), desc="MC Dropout Iterations"
                ):
                    set_seed(mc_dropout_iter)
                    if aleatoric_uncertainty:
                        batch_output[mc_dropout_iter] = trained_model(batch_input)[
                            :, 0
                        ].unsqueeze(1)
                        batch_aleatoric_variance[mc_dropout_iter] = trained_model(
                            batch_input
                        )[:, 1].unsqueeze(1)
                    else:
                        batch_output[mc_dropout_iter] = trained_model(batch_input)
                    if transforms:
                        batch_output[mc_dropout_iter] = transforms.inverse(
                            batch_output[mc_dropout_iter]
                        )
                        if aleatoric_uncertainty:
                            batch_aleatoric_variance[mc_dropout_iter] = (
                                transforms.inverse_variance(
                                    batch_aleatoric_variance[mc_dropout_iter]
                                )
                            )
                batch_epistemic_uncertainty = batch_output.std(
                    dim=0
                )  # * 2  # 2 standard deviations
                batch_output = batch_output.mean(dim=0)
                if aleatoric_uncertainty:
                    batch_aleatoric_uncertainty = torch.sqrt(
                        batch_aleatoric_variance.mean(dim=0)
                    )
            else:
                if aleatoric_uncertainty:
                    batch_output = trained_model(batch_input)[:, 0].unsqueeze(1)
                    batch_aleatoric_variance = trained_model(batch_input)[
                        :, 1
                    ].unsqueeze(1)
                    if transforms:
                        batch_output = transforms.inverse(batch_output)
                        batch_aleatoric_variance = transforms.inverse_variance(
                            batch_aleatoric_variance
                        )
                    batch_aleatoric_uncertainty = torch.sqrt(batch_aleatoric_variance)
                else:
                    batch_output = trained_model(batch_input)[:, 0].unsqueeze(1)
                    if transforms:
                        batch_output = transforms.inverse(batch_output)

            if transforms:
                if not estimating_washout_fraction:
                    batch_input_channel_1 = batch_input[:, 1, :, :, :].unsqueeze(1)
                batch_input = input_transforms.inverse(
                    batch_input[:, 0, :, :, :].unsqueeze(1)
                )
                if not estimating_washout_fraction:
                    batch_input -= min_decay  # removing the physical decay
                    batch_input *= 60  # converting to 1/minutes
                if region_types_flag:
                    if (
                        edges_region_types is None
                    ):  # if bucketized, the tensor already has one channel with an integer values for each region type
                        _, batch_output = torch.max(batch_output, dim=1)
                        batch_output = batch_output.unsqueeze(1)
                    else:
                        batch_output = (
                            torch.bucketize(batch_output, edges_region_types) - 1
                        )
                        batch_output[batch_output == -1] = 0
                        batch_output[batch_output == num_classes] = num_classes - 1
                        batch_output = (
                            num_classes - 1 - batch_output
                        )  # to have the same order as the original region types (which are ordered in ascending order of mean life and in descending order of decay rate)
                    error_output = (
                        ((batch_output - batch_target) == 0).cpu().numpy()
                    )  # 1 if the output is correct, 0 otherwise
                else:
                    batch_target = transforms.inverse(batch_target)
                    if not estimating_washout_fraction:
                        batch_output -= min_decay
                        batch_target -= min_decay
                        batch_output *= 60
                        batch_target *= 60
                        if epistemic_uncertainty:
                            batch_epistemic_uncertainty *= 60
                        if aleatoric_uncertainty:
                            batch_aleatoric_uncertainty *= 60
                    error_input = torch.abs(batch_input - batch_target).cpu().numpy()
                    error_output = torch.abs(batch_output - batch_target).cpu().numpy()
            input = batch_input.cpu().numpy()
            if not estimating_washout_fraction:
                input_channel_1 = batch_input_channel_1.cpu().numpy()
            output = batch_output.cpu().numpy()
            target = batch_target.cpu().numpy()
            CT = batch_CT.cpu().numpy()
            tumor_mask = batch_tumor_mask.cpu().numpy()
            if epistemic_uncertainty:
                epistemic_uncertainty_array = batch_epistemic_uncertainty.cpu().numpy()
            if aleatoric_uncertainty:
                aleatoric_uncertainty_array = batch_aleatoric_uncertainty.cpu().numpy()

            break  # only want one sample here

    if epistemic_uncertainty and not aleatoric_uncertainty:
        uncertainty_array = epistemic_uncertainty_array
    elif aleatoric_uncertainty and not epistemic_uncertainty:
        uncertainty_array = aleatoric_uncertainty_array
    elif epistemic_uncertainty and aleatoric_uncertainty:
        # COMBINING UNCERTAINTIES AS IF THEY WERE INDEPENDENT (common assumption)
        uncertainty_array = np.sqrt(
            epistemic_uncertainty_array**2 + aleatoric_uncertainty_array**2
        )

    vmin_CT = -125
    vmax_CT = 225

    if region_types_flag:
        vmax_error = 1
    else:
        vmax_error = 0.0
        if plane == "coronal" or plane == "y":
            tumor_mask_slice = tumor_mask[0, 0, :, num_slice, :]
            error_input_slice = error_input[0, 0, :, num_slice, :] * tumor_mask_slice
            error_output_slice = error_output[0, 0, :, num_slice, :] * tumor_mask_slice
            if epistemic_uncertainty or aleatoric_uncertainty:
                uncertainty_slice = (
                    uncertainty_array[0, 0, :, num_slice, :] * tumor_mask_slice
                )
        elif plane == "axial" or plane == "z":
            tumor_mask_slice = tumor_mask[0, 0, :, :, num_slice]
            error_input_slice = error_input[0, 0, :, :, num_slice] * tumor_mask_slice
            error_output_slice = error_output[0, 0, :, :, num_slice] * tumor_mask_slice
            if epistemic_uncertainty or aleatoric_uncertainty:
                uncertainty_slice = (
                    uncertainty_array[0, 0, :, :, num_slice] * tumor_mask_slice
                )
        elif plane == "sagital" or plane == "x":
            tumor_mask_slice = tumor_mask[0, 0, num_slice, :, :]
            error_input_slice = error_input[0, 0, num_slice, :, :] * tumor_mask_slice
            error_output_slice = error_output[0, 0, num_slice, :, :] * tumor_mask_slice
            if epistemic_uncertainty or aleatoric_uncertainty:
                uncertainty_slice = (
                    uncertainty_array[0, 0, num_slice, :, :] * tumor_mask_slice
                )
        if no_fit:
            vmax_error = max(np.max(error_output_slice), vmax_error)
        else:
            vmax_error = max(
                np.max(error_input_slice), np.max(error_output_slice), vmax_error
            )
        if epistemic_uncertainty or aleatoric_uncertainty:
            vmax_error = max(vmax_error, np.max(uncertainty_slice))
    if estimating_washout_fraction:
        vmin_main = np.min(target[0][tumor_mask[0] == 1])
        vmax_main = np.max(target[0][tumor_mask[0] == 1])
    else:
        vmin_main = 0.0
        vmax_main = max_decay
    sns.set_style("white")
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams["font.sans-serif"] = "Helvetica"

    if plane == "coronal" or plane == "y":
        CT_slice = np.flip(CT[0, 0, :, num_slice, :].T, axis=0)
        input_slice = np.flip(input[0, 0, :, num_slice, :].T, axis=0)
        if not estimating_washout_fraction:
            input_channel_1_slice = np.flip(
                input_channel_1[0, 0, :, num_slice, :].T, axis=0
            )
        output_slice = np.flip(output[0, 0, :, num_slice, :].T, axis=0)
        target_slice = np.flip(target[0, 0, :, num_slice, :].T, axis=0)
        error_output_slice = np.flip(error_output[0, 0, :, num_slice, :].T, axis=0)
        if not region_types_flag:
            error_input_slice = np.flip(error_input[0, 0, :, num_slice, :].T, axis=0)
        if epistemic_uncertainty or aleatoric_uncertainty:
            uncertainty_slice = np.flip(
                uncertainty_array[0, 0, :, num_slice, :].T, axis=0
            )
        tumor_mask_slice = np.flip(tumor_mask[0, 0, :, num_slice, :].T, axis=0)
    elif plane == "axial" or plane == "z":
        CT_slice = CT[0, 0, :, :, num_slice]
        input_slice = input[0, 0, :, :, num_slice]
        if not estimating_washout_fraction:
            input_channel_1_slice = input_channel_1[0, 0, :, :, num_slice]
        output_slice = output[0, 0, :, :, num_slice]
        target_slice = target[0, 0, :, :, num_slice]
        error_output_slice = error_output[0, 0, :, :, num_slice]
        if not region_types_flag:
            error_input_slice = error_input[0, 0, :, :, num_slice]
        if epistemic_uncertainty or aleatoric_uncertainty:
            uncertainty_slice = uncertainty_array[0, 0, :, :, num_slice]
        tumor_mask_slice = tumor_mask[0, 0, :, :, num_slice]
    elif plane == "sagital" or plane == "x":
        CT_slice = np.flip(CT[0, 0, num_slice, :, :], axis=1).T
        input_slice = np.flip(input[0, 0, num_slice, :, :], axis=1).T
        if not estimating_washout_fraction:
            input_channel_1_slice = np.flip(
                input_channel_1[0, 0, num_slice, :, :], axis=1
            ).T
        output_slice = np.flip(output[0, 0, num_slice, :, :], axis=1).T
        target_slice = np.flip(target[0, 0, num_slice, :, :], axis=1).T
        error_output_slice = np.flip(error_output[0, 0, num_slice, :, :], axis=1).T
        if not region_types_flag:
            error_input_slice = np.flip(error_input[0, 0, num_slice, :, :], axis=1).T
        if epistemic_uncertainty or aleatoric_uncertainty:
            uncertainty_slice = np.flip(
                uncertainty_array[0, 0, num_slice, :, :], axis=1
            ).T
        tumor_mask_slice = np.flip(tumor_mask[0, 0, num_slice, :, :], axis=1).T
    else:
        raise ValueError("Plane must be coronal, sagittal or axial")

    CT_slice = np.squeeze(CT_slice)
    input_slice = np.squeeze(input_slice)
    if not estimating_washout_fraction:
        input_channel_1_slice = np.squeeze(input_channel_1_slice)
    output_slice = np.squeeze(output_slice)
    target_slice = np.squeeze(target_slice)
    error_output_slice = np.squeeze(error_output_slice)
    tumor_mask_slice = np.squeeze(tumor_mask_slice)
    if not region_types_flag:
        error_input_slice = np.squeeze(error_input_slice)
    if epistemic_uncertainty or aleatoric_uncertainty:
        uncertainty_slice = np.squeeze(uncertainty_slice)

    # Since we are only interested in the error inside the tumor, we will mask the error maps with the tumor mask
    if not region_types_flag:
        error_input_slice = error_input_slice * tumor_mask_slice
        error_output_slice = error_output_slice * tumor_mask_slice
        if epistemic_uncertainty or aleatoric_uncertainty:
            uncertainty_slice = uncertainty_slice * tumor_mask_slice
    else:
        error_output_slice[tumor_mask_slice == 0] = (
            True  # only the tumor region is considered for the error
        )
        output_slice[tumor_mask_slice == 0] = target_slice[tumor_mask_slice == 0]

    # Plot the maps with a transparent background outside the tumor
    save_plot_dir = os.path.join(save_plot_dir, "tumor_maps")
    if not os.path.exists(save_plot_dir):
        os.makedirs(save_plot_dir)

    # CT
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis("off")
    norm = Normalize(vmin=vmin_CT, vmax=vmax_CT, clip=True)
    CT_slice_normalized = norm(CT_slice)
    img_rgba = plt.cm.gray(CT_slice_normalized)
    img_rgba[..., -1] = tumor_mask_slice.astype(np.float32)
    ax.imshow(img_rgba, interpolation="none")
    fig.savefig(
        os.path.join(save_plot_dir, "CT_slice.png"),
        dpi=600,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)

    if region_types_flag:
        # Estimated region type maps
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.axis("off")
        vmin = 0
        vmax = num_classes - 1
        norm_region_types = Normalize(vmin=vmin, vmax=vmax, clip=True)
        output_slice = norm_region_types(output_slice)
        img_rgba = cmap_region_types(output_slice)
        img_rgba[..., -1] = tumor_mask_slice.astype(np.float32)
        ax.imshow(img_rgba, interpolation="none")
        fig.savefig(
            os.path.join(save_plot_dir, "region_types_slice.png"),
            dpi=600,
            bbox_inches="tight",
            transparent=True,
        )

        # Target region type maps
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.axis("off")
        target_slice = norm_region_types(target_slice)
        img_rgba = cmap_region_types(target_slice)
        img_rgba[..., -1] = tumor_mask_slice.astype(np.float32)
        ax.imshow(img_rgba, interpolation="none")
        fig.savefig(
            os.path.join(save_plot_dir, "target_region_types_slice.png"),
            dpi=600,
            bbox_inches="tight",
            transparent=True,
        )

    else:
        if not no_fit:
            norm = Normalize(vmin=vmin_main, vmax=vmax_main, clip=True)
            # Corrected slow washout / washout fraction
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.axis("off")
            output_slice = norm(output_slice)
            img_rgba = plt.cm.inferno(output_slice)
            img_rgba[..., -1] = tumor_mask_slice.astype(np.float32)
            ax.imshow(img_rgba, interpolation="none")
            if estimating_washout_fraction:
                fig.savefig(
                    os.path.join(save_plot_dir, "corrected_washout_fraction_slice.png"),
                    dpi=600,
                    bbox_inches="tight",
                    transparent=True,
                )
            else:
                fig.savefig(
                    os.path.join(
                        save_plot_dir, "corrected_slow_washout_rate_slice.png"
                    ),
                    dpi=600,
                    bbox_inches="tight",
                    transparent=True,
                )
            plt.close(fig)

            # Uncorrected slow washout
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.axis("off")
            input_slice = norm(input_slice)
            img_rgba = plt.cm.inferno(input_slice)
            img_rgba[..., -1] = tumor_mask_slice.astype(np.float32)
            ax.imshow(img_rgba, interpolation="none")
            if estimating_washout_fraction:
                fig.savefig(
                    os.path.join(
                        save_plot_dir, "uncorrected_washout_fraction_slice.png"
                    ),
                    dpi=600,
                    bbox_inches="tight",
                    transparent=True,
                )
            else:
                fig.savefig(
                    os.path.join(
                        save_plot_dir, "uncorrected_slow_washout_rate_slice.png"
                    ),
                    dpi=600,
                    bbox_inches="tight",
                    transparent=True,
                )
            plt.close(fig)

            # uncertainty map
            if epistemic_uncertainty or aleatoric_uncertainty:
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.axis("off")
                norm_error = Normalize(vmin=0, vmax=vmax_error, clip=True)
                uncertainty_slice = norm_error(uncertainty_slice)
                img_rgba = plt.cm.Reds(uncertainty_slice)
                img_rgba[..., -1] = tumor_mask_slice.astype(np.float32)
                ax.imshow(img_rgba, interpolation="none")
                if estimating_washout_fraction:
                    fig.savefig(
                        os.path.join(
                            save_plot_dir, "washout_fraction_uncertainty_slice.png"
                        ),
                        dpi=600,
                        bbox_inches="tight",
                        transparent=True,
                    )
                else:
                    fig.savefig(
                        os.path.join(
                            save_plot_dir, "slow_washout_rate_uncertainty_slice.png"
                        ),
                        dpi=600,
                        bbox_inches="tight",
                        transparent=True,
                    )
                plt.close(fig)

            if not estimating_washout_fraction:
                # Target slow washout
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.axis("off")
                target_slice = norm(target_slice)
                img_rgba = plt.cm.inferno(target_slice)
                img_rgba[..., -1] = tumor_mask_slice.astype(np.float32)
                ax.imshow(img_rgba, interpolation="none")
                fig.savefig(
                    os.path.join(save_plot_dir, "target_slow_washout_rate_slice.png"),
                    dpi=600,
                    bbox_inches="tight",
                    transparent=True,
                )
                plt.close(fig)

                # Initial activity (channel 1)
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.axis("off")
                img_rgba = plt.cm.summer(input_channel_1_slice)
                img_rgba[..., -1] = tumor_mask_slice.astype(np.float32)
                ax.imshow(img_rgba, interpolation="none")
                fig.savefig(
                    os.path.join(save_plot_dir, "initial_activity_slice.png"),
                    dpi=600,
                    bbox_inches="tight",
                    transparent=True,
                )


def sparsification_error_batched(flat_uncertainty, flat_output, flat_target):
    """
    Compute the sparsification error for each element in the batch by
    removing pixels in descending order of uncertainty and measuring
    the error on the remaining pixels. If the uncertainty estimate is good,
    the error should decrease monotonically as pixels are removed.
    """
    batch_errors_list = []
    removed_pixels_list = []
    step = 20  # only compute the error every "step" pixels removed
    num_pixels = flat_output.numel()
    print(f"Number of pixels: {num_pixels}\n")
    # Sort the pixels by uncertainty in descending order
    if num_pixels > 0:  # Ensure the array is not empty
        sorted_indices = flat_uncertainty.argsort(descending=True)
        # sort the output and target tensors in the same order
        flat_output = flat_output[sorted_indices]
        flat_target = flat_target[sorted_indices]
    else:
        raise ValueError("Array is empty. Cannot sort or process.")
    # Compute the error for each sparsification level
    batch_errors_list.append(torch.median(torch.abs(flat_output - flat_target)).item())
    removed_pixels_list.append(0.0)
    for start_idx in tqdm(
        range(0, num_pixels, step), desc="Computing Sparsification Error"
    ):
        # Identify indices for the current chunk of step pixels
        flat_output = flat_output[step:]
        flat_target = flat_target[step:]
        remaining_pixels = flat_target.numel()
        if remaining_pixels > step:
            batch_errors_list.append(
                torch.median(torch.abs(flat_output - flat_target)).item()
            )
            removed_pixels_list.append((start_idx + step) / num_pixels * 100)
    return torch.tensor(batch_errors_list), torch.tensor(removed_pixels_list)


def save_model_complexity(
    model, img_size, model_name="undefined", model_sizes_txt="models/model_sizes.txt"
):
    """Save the model complexity information (FLOPs and parameters) to a text file."""
    flops, params = get_model_complexity_info(
        model, img_size, as_strings=False, print_per_layer_stat=False
    )
    complexity_info = f"\n{model_name}: {2 * flops / 1e9:.2e} GFLOPs, {params:.2e} parameters"  # Append the information to the file

    with open(model_sizes_txt, "a") as file:
        file.write(complexity_info)
    return None
