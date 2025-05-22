''' This file contains the testing function for the model. It tests the model on the test set 
and saves the results to a text file and corresponding plots.'''

import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import wilcoxon
from utils_model import sparsification_error_batched

script_dir = os.path.dirname(os.path.abspath(__file__))
# font_path = os.path.join(script_dir, '../images/Times_New_Roman.ttf')
font_path = os.path.join(script_dir, "../images/Helvetica.ttf")
font_manager.fontManager.addfont(font_path)
# plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["font.sans-serif"] = "Helvetica"


def test(
    trained_model,
    test_loader,
    device,
    results_dir=".",
    transforms=None,
    input_transforms=None,
    save_plot_dir=None,
    voxel_size=(1.9531, 1.9531, 1.5),
    min_decay=0.0,  # minimum decay rate, will correspond with physical decay
    max_decay=1.0,  # maximum decay rate, will correspond with most perfused tissue
    region_types_flag=False,
    num_classes=1,
    edges_region_types=None,  # to convert output decay map to region types
    epistemic_uncertainty=False,  # If True, calculate the epistemic uncertainty with monte carlo dropout
    mc_dropout_iters=30,
    aleatoric_uncertainty=False,
    estimating_washout_fraction=False,
    no_fit=False,  # the model is trained using the five PET frames rather than the Uncorrected washout map
):
    """Test the trained model on the test set, providing metrics and plots."""
    if input_transforms is None:
        input_transforms = transforms

    # Test loop (after the training is complete)
    time_list = []

    if estimating_washout_fraction:
        max_decay *= 100  # convert to percentage
        min_decay *= 100
    else:
        max_decay *= 60  # convert to 1/minutes
        min_decay *= 60

    if region_types_flag:
        loss_function = nn.CrossEntropyLoss()
        region_pred_accuracy = (
            []
        )  # store the percentage of misclassified pixels for each region
        region_pred_before_accuracy = (
            []
        )  # store the percentage of misclassified pixels for each region before the denoising model is applied
    else:
        loss_function = nn.L1Loss(reduction="none")
        ssim_loss = StructuralSimilarityIndexMeasure(
            data_range=max_decay - min_decay
        ).to(
            device
        )  # data_range is the range of the data, in this case, the range of the BIOLOGICAL decay rate (if it was the whole decay rate, it would go from min_decay to max_decay)
        # Input errors
        input_error_list = []  # Decay rate error for input (Uncorrected)
        region_input_error_list = (
            []
        )  # Decay rate error for each region in the input (Uncorrected)
        ssim_input_list = []  # Structural similarity index for input (Uncorrected)
        region_ssim_input_list = (
            []
        )  # Structural similarity index for each region in the input (Uncorrected)

        # Output errors
        output_error_list = []  # Decay rate error for output (corrected)
        region_output_error_list = (
            []
        )  # Decay rate error for each region in the output (corrected)
        ssim_output_list = []  # Structural similarity index for output (corrected)
        region_ssim_output_list = (
            []
        )  # Structural similarity index for each region in the output (corrected)

    # Uncertainty
    total_uncertainty_list = []  # store the total uncertainty to plot against the error
    remaining_pixels_list = (
        []
    )  # store the remaining pixels for every combination of remaining pixels and error across all the images
    sparsification_error_list = (
        []
    )  # store the sparsification error for every combination of remaining pixels and error across all the images

    region_sizes = []
    ssim_region_sizes = []
    voxel_size = np.array(voxel_size) / 10  # Convert to cm
    voxel_volume = np.prod(voxel_size)

    with torch.no_grad():
        for batch_input, batch_target, batch_regions, batch_tumor_mask in tqdm(
            test_loader, desc="Testing batch"
        ):
            batch_input = batch_input.to(device)
            batch_input = batch_input[:, :-1, :, :, :]
            batch_target = batch_target.to(device)
            batch_tumor_mask = batch_tumor_mask.to(torch.float32).to(device)
            start_time = time.time()
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
                batch_epistemic_uncertainty = batch_output.std(dim=0)
                batch_output = batch_output.mean(dim=0)
                if aleatoric_uncertainty:  # Combine aleatoric and epistemic uncertainty
                    batch_aleatoric_uncertainty = torch.sqrt(
                        batch_aleatoric_variance.mean(dim=0)
                    )
                    batch_uncertainty = torch.sqrt(
                        batch_aleatoric_uncertainty**2 + batch_epistemic_uncertainty**2
                    )
                time_list.append(
                    (time.time() - start_time)
                    * 1000
                    / batch_input.shape[0]
                    / mc_dropout_iters
                )  # time per loading per sample per mc dropout iteration in ms

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
                    batch_uncertainty = torch.sqrt(batch_aleatoric_variance)
                else:
                    batch_uncertainty = [None] * batch_input.shape[0]
                    batch_output = trained_model(batch_input)  # generating images
                    if transforms:
                        batch_output = transforms.inverse(batch_output)
                time_list.append(
                    (time.time() - start_time) * 1000 / batch_input.shape[0]
                )  # time per loading in ms

            batch_input = batch_input[:, 0, ...].unsqueeze(
                1
            )  # only keeping the washout rate channel
            if transforms:
                batch_input = input_transforms.inverse(
                    batch_input[:, 0, ...].unsqueeze(1)
                )
                if region_types_flag:
                    if edges_region_types is None:
                        batch_predictions = torch.argmax(batch_output, dim=1).unsqueeze(
                            1
                        )
                        batch_predictions_before = torch.argmax(
                            batch_input, dim=1
                        ).unsqueeze(1)
                    else:
                        batch_input = (
                            torch.bucketize(batch_input, edges_region_types) - 1
                        )
                        batch_output = (
                            torch.bucketize(batch_output, edges_region_types) - 1
                        )
                        batch_target = (
                            torch.bucketize(batch_target, edges_region_types) - 1
                        )
                        batch_input[batch_input == -1] = 0
                        batch_output[batch_output == -1] = 0
                        batch_target[batch_target == -1] = 0
                        batch_input[batch_input > num_classes - 1] = num_classes - 1
                        batch_output[batch_output > num_classes - 1] = num_classes - 1
                        batch_target[batch_target > num_classes - 1] = num_classes - 1

                        batch_predictions_before = batch_input.clone()
                        batch_predictions = batch_output.clone()
                        batch_predictions_before = batch_predictions_before.to(
                            torch.int8
                        )
                        batch_predictions = batch_predictions.to(torch.int8)
                        batch_target = batch_target.to(torch.long)

                        # one hot encode
                        batch_input = (
                            torch.nn.functional.one_hot(
                                batch_input.to(torch.long), num_classes=num_classes
                            )
                            .permute(0, 5, 1, 2, 3, 4)
                            .to(torch.float32)
                        )
                        batch_output = (
                            torch.nn.functional.one_hot(
                                batch_output.to(torch.long), num_classes=num_classes
                            )
                            .permute(0, 5, 1, 2, 3, 4)
                            .to(torch.float32)
                        )

                        batch_input = batch_input.squeeze(2)
                        batch_output = batch_output.squeeze(2)
                else:
                    batch_target = transforms.inverse(batch_target)
                    batch_predictions = [None] * batch_input.shape[0]
                    batch_predictions_before = [None] * batch_input.shape[0]
                    if estimating_washout_fraction:
                        batch_input *= 100
                        batch_output *= 100
                        batch_target *= 100
                        if epistemic_uncertainty or aleatoric_uncertainty:
                            batch_uncertainty *= 100
                    else:
                        batch_input *= 60  # converting to 1/minutes
                        batch_output *= 60
                        batch_target *= 60
                        batch_input -= min_decay  # removing the physical decay
                        batch_output -= min_decay
                        batch_target -= min_decay
                        if epistemic_uncertainty or aleatoric_uncertainty:
                            batch_uncertainty *= 60
            else:
                batch_input = batch_input[:, 0, ...].unsqueeze(
                    1
                )  # removing the CT channel

            # Calculate the errors
            # batch_input = batch_input * batch_tumor_mask
            # batch_target = batch_target * batch_tumor_mask
            # batch_output = batch_output * batch_tumor_mask
            # batch_uncertainty = batch_uncertainty * batch_tumor_mask

            if not region_types_flag:
                input_error_list.append(
                    loss_function(
                        batch_input[batch_tumor_mask.to(torch.bool)],
                        batch_target[batch_tumor_mask.to(torch.bool)],
                    )
                )
                output_error_list.append(
                    loss_function(
                        batch_output[batch_tumor_mask.to(torch.bool)],
                        batch_target[batch_tumor_mask.to(torch.bool)],
                    )
                )
                if epistemic_uncertainty or aleatoric_uncertainty:
                    total_uncertainty_list.append(
                        batch_uncertainty[batch_tumor_mask.to(torch.bool)]
                    )
                    # sparsification_error_batch, remaining_pixels_batch = sparsification_error(batch_uncertainty, batch_output, batch_target, batch_tumor_mask)
                    sparsification_error_batch, remaining_pixels_batch = (
                        sparsification_error_batched(
                            batch_uncertainty[batch_tumor_mask.to(torch.bool)],
                            batch_output[batch_tumor_mask.to(torch.bool)],
                            batch_target[batch_tumor_mask.to(torch.bool)],
                        )
                    )

                    sparsification_error_list.append(sparsification_error_batch)
                    remaining_pixels_list.append(remaining_pixels_batch)

            # Region sizes vs errors
            for (
                image_input,
                image_output,
                image_target,
                image_regions,
                image_predictions,
                image_predictions_before,
                image_tumor_mask,
                image_uncertainty,
            ) in zip(
                batch_input,
                batch_output,
                batch_target,
                batch_regions,
                batch_predictions,
                batch_predictions_before,
                batch_tumor_mask,
                batch_uncertainty,
            ):
                image_input = image_input.squeeze()
                image_target = image_target.squeeze()
                image_output = image_output.squeeze()
                image_tumor_mask = image_tumor_mask.squeeze()
                nonzero_indices = torch.nonzero(image_tumor_mask)
                x_min, y_min, z_min = nonzero_indices.min(dim=0)[0]
                x_max, y_max, z_max = nonzero_indices.max(dim=0)[0]
                if region_types_flag:
                    image_uncertainty = image_uncertainty.squeeze()
                    image_predictions = image_predictions.squeeze()
                    image_predictions_before = image_predictions_before.squeeze()
                else:
                    ssim_input_list.append(
                        ssim_loss(
                            image_input[
                                x_min:x_max, y_min:y_max, z_min:z_max
                            ].unsqueeze(0),
                            image_target[
                                x_min:x_max, y_min:y_max, z_min:z_max
                            ].unsqueeze(0),
                        )
                    )
                    ssim_output_list.append(
                        ssim_loss(
                            image_output[
                                x_min:x_max, y_min:y_max, z_min:z_max
                            ].unsqueeze(0),
                            image_target[
                                x_min:x_max, y_min:y_max, z_min:z_max
                            ].unsqueeze(0),
                        )
                    )

                background_region = (
                    image_regions[-1].to(device) & image_tumor_mask.to(torch.bool)
                ).to(
                    torch.bool
                )  # image_regions[-1] is everything that is not in the regions, incluing the empty background outside the tumor
                image_regions[-1] = background_region

                # Region sizes vs erros
                for _, region in enumerate(image_regions):
                    region_size = region.sum().item() * voxel_volume
                    region_sizes.extend([region_size] * region.sum().item())
                    nonzero_indices = torch.nonzero(region)
                    x_min, y_min, z_min = nonzero_indices.min(dim=0)[0]
                    x_max, y_max, z_max = nonzero_indices.max(dim=0)[0]
                    if region_types_flag:
                        region_pred_accuracy.append(
                            (image_target[region] == image_predictions[region]).to(
                                torch.float16
                            )
                        )
                        region_pred_before_accuracy.append(
                            (
                                image_target[region] == image_predictions_before[region]
                            ).to(torch.float16)
                        )
                        if region_size > 0.0:  # if region larger than x mL
                            total_uncertainty_list.append(
                                image_uncertainty[region].view(-1).cpu()
                            )
                    else:
                        region_input_error_list.append(
                            torch.abs(image_input[region] - image_target[region])
                        )
                        region_output_error_list.append(
                            torch.abs(image_output[region] - image_target[region])
                        )
                        if (
                            x_max - x_min > 5
                            and y_max - y_min > 5
                            and z_max - z_min > 5
                            and not region_types_flag
                        ):  # ssim won't work if the region is too small
                            ssim_region_sizes.append(region_size)
                            region_ssim_input_list.append(
                                ssim_loss(
                                    image_input[
                                        x_min:x_max, y_min:y_max, z_min:z_max
                                    ].unsqueeze(0),
                                    image_target[
                                        x_min:x_max, y_min:y_max, z_min:z_max
                                    ].unsqueeze(0),
                                )
                            )
                            region_ssim_output_list.append(
                                ssim_loss(
                                    image_output[
                                        x_min:x_max, y_min:y_max, z_min:z_max
                                    ].unsqueeze(0),
                                    image_target[
                                        x_min:x_max, y_min:y_max, z_min:z_max
                                    ].unsqueeze(0),
                                )
                            )

    if not region_types_flag:
        input_error_list = torch.cat(input_error_list)
        ssim_input_list = torch.tensor(ssim_input_list)
        output_error_list = torch.cat(output_error_list)
        ssim_output_list = torch.tensor(ssim_output_list)
        if epistemic_uncertainty or aleatoric_uncertainty:
            total_uncertainty_list = torch.cat(total_uncertainty_list)
            remaining_pixels_list = torch.cat(remaining_pixels_list)
            sparsification_error_list = torch.cat(sparsification_error_list)
        if not estimating_washout_fraction:
            text_results = (
                "Corrected output vs target:\n"
                f"Washout Rate Median Absolute Error (min⁻¹): {torch.median(output_error_list):.6f} +- {torch.quantile(output_error_list, 0.75) - torch.quantile(output_error_list, 0.25):.6f}\n"
                f"SSIM: {torch.mean(ssim_output_list):.4f} +- {torch.std(ssim_output_list):.4f}\n"
                f"\nTime per loading (ms): {np.mean(np.array(time_list)):.4f} +- {np.std(np.array(time_list)):.4f}"
            )
            if not no_fit:
                # if directly estimating the washout map from the PET frames, there is no Uncorrected input
                text_results = (
                    "Uncorrected input vs target:\n"
                    f"Washout Rate Median Absolute Error (min⁻¹): {torch.median(input_error_list):.6f} +- {torch.quantile(input_error_list, 0.75) - torch.quantile(input_error_list, 0.25):.6f}\n"
                    f"SSIM: {torch.mean(ssim_input_list):.4f} +- {torch.std(ssim_input_list):.4f}\n"
                    f"\n\n" + text_results
                )
            if epistemic_uncertainty:
                total_uncertainty_list = torch.tensor(total_uncertainty_list)
                text_results += f"\n\nWashout Rate Uncertainty (std) (min⁻¹): {torch.median(total_uncertainty_list):.4f} +- {torch.quantile(total_uncertainty_list, 0.75) - torch.quantile(total_uncertainty_list, 0.25):.4f}"

        else:
            text_results = (
                "Washout Fraction Results"
                "Uncorrected input vs target:\n"
                f"Median Absolute Error (%): {torch.median(input_error_list):.6f} +- {torch.quantile(input_error_list, 0.75) - torch.quantile(input_error_list, 0.25):.6f}\n"
                f"SSIM: {torch.mean(ssim_input_list):.4f} +- {torch.std(ssim_input_list):.4f}\n"
                f"\n\nCorrected output vs target:\n"
                f"Median Absolute Error (%) {torch.median(output_error_list):.6f} +- {torch.quantile(output_error_list, 0.75) - torch.quantile(output_error_list, 0.25):.6f}\n"
                f"SSIM: {torch.mean(ssim_output_list):.4f} +- {torch.std(ssim_output_list):.4f}\n"
                f"\nTime per loading (ms): {np.mean(np.array(time_list)):.4f} +- {np.std(np.array(time_list)):.4f}"
            )

            if epistemic_uncertainty:
                total_uncertainty_list = torch.tensor(total_uncertainty_list)
                text_results += f"\n\n Uncertainty (std): {torch.median(total_uncertainty_list):.4f} +- {torch.quantile(total_uncertainty_list, 0.75) - torch.quantile(total_uncertainty_list, 0.25):.4f}"

    if save_plot_dir:
        font_size = 25
        sns.set_theme(style="whitegrid", context="talk", font="Helvetica")
        colorblind_palette = sns.color_palette("colorblind")
        num_bins = 10
        bins = np.array(region_sizes).max() * np.linspace(0, 1, num_bins + 1)
        label_region_sizes = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        labels = [f"{label_region_sizes[i]:.1f}" for i in range(len(bins) - 1)]
        if no_fit:
            value_vars = ["Corrected"]
            palette = {"Corrected": colorblind_palette[1]}
        else:
            value_vars = ["Uncorrected", "Corrected"]
            palette = {
                "Corrected": colorblind_palette[1],
                "Uncorrected": colorblind_palette[0],
            }
        if region_types_flag:
            # Plot error per region
            region_pred_accuracy = (
                torch.cat(region_pred_accuracy).to(int).cpu().numpy() * 100
            )  # Convert to percentage
            total_uncertainty = torch.cat(total_uncertainty_list).numpy()
            df = pd.DataFrame(
                {
                    "Region Volume (ml)": region_sizes,
                    "Corrected": region_pred_accuracy,
                    "Uncertainty": total_uncertainty,
                }
            )
            if not no_fit:
                region_pred_before_accuracy = (
                    torch.cat(region_pred_before_accuracy).to(int).cpu().numpy() * 100
                )
                df["Uncorrected"] = region_pred_before_accuracy
            df["Region Volume (ml)"] = pd.cut(
                df["Region Volume (ml)"], bins=bins, labels=labels
            )

            accuracy_df = pd.melt(
                df,
                id_vars=["Region Volume (ml)"],
                value_vars=value_vars,
                var_name="Type",
                value_name="Accuracy (%)",
            )

            # Remove the top 10% of values with most uncertainty
            cut90 = df["Uncertainty"].quantile(0.90)
            cut80 = df["Uncertainty"].quantile(0.80)
            cutoff_90_label = "Corrected (Top 10% Uncertain Voxels Removed)"
            cutoff_80_label = "Corrected (Top 20% Uncertain Voxels Removed)"

            # melt only Corrected for each cutoff
            m90 = pd.melt(
                df[df["Uncertainty"] < cut90],
                id_vars=["Region Volume (ml)"],
                value_vars=["Corrected"],
                var_name="Type",
                value_name="Accuracy (%)",
            )
            m90["Type"] = cutoff_90_label

            m80 = pd.melt(
                df[df["Uncertainty"] < cut80],
                id_vars=["Region Volume (ml)"],
                value_vars=["Corrected"],
                var_name="Type",
                value_name="Accuracy (%)",
            )
            m80["Type"] = cutoff_80_label

            # merge the dataframes
            mean_values = pd.concat([accuracy_df, m90, m80], ignore_index=True)
            mean_values = (
                mean_values.groupby(["Region Volume (ml)", "Type"])["Accuracy (%)"]
                .mean()
                .reset_index()
            )

            palette_accuracy = {
                "Uncorrected": colorblind_palette[0],
                "Corrected": colorblind_palette[1],
                cutoff_90_label: colorblind_palette[2],
                cutoff_80_label: colorblind_palette[3],
            }
            palette_accuracy = {
                k: v
                for k, v in palette_accuracy.items()
                if k in mean_values["Type"].unique()
            }
            hue_order = ["Uncorrected", "Corrected", cutoff_90_label, cutoff_80_label]
            hue_order = [h for h in hue_order if h in mean_values["Type"].unique()]

            fig, ax = plt.subplots(figsize=(12, 7))
            sns.barplot(
                x="Region Volume (ml)",
                y="Accuracy (%)",
                hue="Type",
                data=mean_values,
                palette=palette_accuracy,
                hue_order=hue_order,
                ax=ax,
            )

            # remove the automatic axes legend
            if ax.get_legend() is not None:
                ax.get_legend().remove()

            plt.ylabel(
                "Tumor Voxel Type\nClassification Accuracy (%)", fontsize=font_size
            )
            plt.xlabel("Region Volume (ml)", fontsize=font_size)
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            # plt.legend(title='Type', title_fontsize=font_size - 5, fontsize=font_size- 5,
            #             loc='upper left')
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=len(labels) // 2,
            )
            # plt.tight_layout()
            fig.subplots_adjust(top=0.86)

            plt.savefig(
                os.path.join(
                    save_plot_dir, "accuracy-vs-size-region-classification.pdf"
                ),
                format="pdf",
                bbox_inches="tight",
            )

        else:
            # Plot boxplot of error vs region volume
            region_output_error_list = torch.cat(region_output_error_list).cpu().numpy()
            df = pd.DataFrame(
                {
                    "Region Volume (ml)": region_sizes,
                    "Corrected": region_output_error_list,
                }
            )
            if not no_fit:
                region_input_error_list = (
                    torch.cat(region_input_error_list).cpu().numpy()
                )
                df["Uncorrected"] = region_input_error_list

            df["Region Volume (ml)"] = pd.cut(
                df["Region Volume (ml)"], bins=bins, labels=labels
            )
            error_df = pd.melt(
                df,
                id_vars=["Region Volume (ml)"],
                value_vars=value_vars,
                var_name="Type",
                value_name="Absolute Error",
            )
            plt.figure(figsize=(12, 7))
            sns.boxplot(
                x="Region Volume (ml)",
                y="Absolute Error",
                hue="Type",
                data=error_df,
                palette=palette,
                showfliers=False,
            )
            if estimating_washout_fraction:
                plt.ylabel("Washout Fraction Abs. Err. (%)", fontsize=font_size)
            else:
                plt.ylabel(
                    r" Washout Rate $\lambda_{B}$ Abs. Err. (min$^{-1}$)",
                    fontsize=font_size,
                )
            plt.xlabel("Region Volume (ml)", fontsize=font_size)
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.legend(title="Type", title_fontsize=font_size, fontsize=font_size)
            plt.tight_layout()
            if estimating_washout_fraction:
                plt.savefig(
                    os.path.join(save_plot_dir, "error-vs-size-washout-fraction.pdf"),
                    format="pdf",
                    bbox_inches="tight",
                )
            else:
                plt.savefig(
                    os.path.join(save_plot_dir, "error-vs-size-washout-rate.pdf"),
                    format="pdf",
                    bbox_inches="tight",
                )

            # save results to text file
            text_results += "\n\nRegion Volume vs Error:\n"
            text_results += "Region Volume (ml) | Uncorrected Mean | Uncorrected Std | Corrected Mean | Corrected Std | p-value\n"
            text_results += "-" * 100 + "\n"
            for region_vol in labels:
                # Get data for this region volume
                region_data = df[df["Region Volume (ml)"] == region_vol]

                # Extract uncorrected and corrected errors
                corrected = region_data["Corrected"].values
                corr_mean = np.mean(corrected)
                corr_std = np.std(corrected)

                if not no_fit:
                    uncorrected = region_data["Uncorrected"].values
                    uncorr_mean = np.mean(uncorrected)
                    uncorr_std = np.std(uncorrected)
                    # Perform statistical test (Wilcoxon signed-rank test for paired samples)
                    _, p_value = wilcoxon(uncorrected, corrected)
                else:
                    uncorr_mean = np.nan
                    uncorr_std = np.nan
                    p_value = np.nan

                # Append to output string
                text_results += f"{region_vol:<16} | {uncorr_mean:.6f}      | {uncorr_std:.6f}       | {corr_mean:.6f}     | {corr_std:.6f}    | {p_value:.10f}\n"

            # Plot boxplot of ssim vs region volume
            bins_ssim = np.array(ssim_region_sizes).max() * np.linspace(
                0, 1, num_bins + 1
            )
            label_region_sizes_ssim = [
                (bins_ssim[i] + bins_ssim[i + 1]) / 2 for i in range(len(bins_ssim) - 1)
            ]
            labels_ssim = [
                f"{label_region_sizes_ssim[i]:.1f}" for i in range(len(bins_ssim) - 1)
            ]
            region_ssim_output_list = (
                torch.tensor(region_ssim_output_list).flatten().cpu().numpy()
            )
            df = pd.DataFrame(
                {
                    "Region Volume (ml)": ssim_region_sizes,
                    "Corrected": region_ssim_output_list,
                }
            )
            if not no_fit:
                region_ssim_input_list = (
                    torch.tensor(region_ssim_input_list).flatten().cpu().numpy()
                )
                df["Uncorrected"] = region_ssim_input_list

            df["Region Volume (ml)"] = pd.cut(
                df["Region Volume (ml)"], bins=bins_ssim, labels=labels_ssim
            )
            error_df = pd.melt(
                df,
                id_vars=["Region Volume (ml)"],
                value_vars=value_vars,
                var_name="Type",
                value_name="SSIM",
            )
            plt.figure(figsize=(12, 7))
            sns.boxplot(
                x="Region Volume (ml)",
                y="SSIM",
                hue="Type",
                data=error_df,
                palette=palette,
                showfliers=False,
            )
            plt.ylabel("SSIM", fontsize=font_size)
            plt.xlabel("Region Volume (ml)", fontsize=font_size)
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.legend(title="Type", title_fontsize=font_size, fontsize=font_size)
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_plot_dir, "ssim-vs-size.pdf"),
                format="pdf",
                bbox_inches="tight",
            )

            if aleatoric_uncertainty or epistemic_uncertainty:
                # Plot Sparsification error
                flat_sparsification_error_list = (
                    sparsification_error_list.flatten().cpu().numpy()
                )
                flat_remaining_pixels_list = (
                    remaining_pixels_list.flatten().cpu().numpy()
                )

                # Define bins for x
                bin_width = 10  # 10% bins
                bins = np.arange(0, 110, bin_width)
                bin_indices = np.digitize(flat_remaining_pixels_list, bins)

                plt.figure(figsize=(12, 7))

                means = [
                    (
                        np.mean(flat_sparsification_error_list[bin_indices == i])
                        if len(flat_sparsification_error_list[bin_indices == i]) > 0
                        else 0
                    )
                    for i in range(1, len(bins))
                ]
                bin_labels = [
                    f"{int(bins[i])}-{int(bins[i+1])}%" for i in range(len(bins) - 1)
                ]
                plt.bar(bin_labels, means, width=0.8, align="center")
                plt.xlabel(
                    "Removed voxels (%)\n(removed in descending order of uncertainty)",
                    fontsize=font_size,
                )
                # incline the x-axis labels
                plt.xticks(fontsize=font_size, rotation=30)
                plt.yticks(fontsize=font_size)
                if estimating_washout_fraction:
                    plt.ylabel("MedAE " + r" (%)", fontsize=font_size)
                else:
                    plt.ylabel("MedAE " + r" (min$^{-1}$)", fontsize=font_size)
                plt.tight_layout()

                plt.savefig(
                    os.path.join(save_plot_dir, "sparsification-error-plot.pdf"),
                    format="pdf",
                    bbox_inches="tight",
                )
    # Save to file
    if not region_types_flag:
        with open(results_dir, "w", encoding="utf-8") as file:
            file.write(text_results)

        print(text_results)

    return None
