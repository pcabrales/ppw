""" This file contains the training function for the model. 
It saves the model with the lowest validation loss and plots the training and validation losses. 
It also saves the losses and the training time. """

import csv
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from monai.losses import FocalLoss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def NLL_loss(output_mean, output_var, target, beta=0.0):
    """Negative log-likelihood loss for the mean and variance of the output,
    penalizing wrong variance predictions as well as wrong mean predictions.
    beta is the variance-weighting term exponent
    """
    output_var_detached = (
        output_var.detach()
    )  # Detach output_var to stop gradients from flowing through it for the beta term
    return torch.mean(
        output_var_detached**beta
        * 0.5
        * (torch.log(output_var) + (target - output_mean) ** 2 / output_var)
    )


def plot_losses(training_losses, val_losses, save_plot_file):
    """Plot the training and validation losses and save the plot"""
    data = {
        "Epoch": range(len(training_losses)),
        "Training Loss": training_losses,
        "Validation Loss": val_losses,
    }
    df = pd.DataFrame(data)
    df = pd.melt(
        df,
        id_vars=["Epoch"],
        value_vars=["Training Loss", "Validation Loss"],
        var_name="Type",
        value_name="Loss",
    )
    sns.set()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Epoch", y="Loss", hue="Type", data=df)
    if min(val_losses) > 0 and min(training_losses) > 0:
        plt.yscale("log")
        plt.ylabel("Loss (log scale)")
    else:
        plt.yscale("linear")
        plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.savefig(save_plot_file, dpi=300, bbox_inches="tight")
    return None


def train(
    model,
    train_loader,
    val_loader,
    epochs=10,
    patience=20,
    learning_rate=1e-4,
    model_dir=".",
    timing_dir=".",
    save_plot_dir=".",
    losses_dir=".",
    accumulation_steps=1,
    region_types_flag=False,
    aleatoric_uncertainty=False,
):
    """Train the model using the training and validation loaders."""
    start_time = time.time()  # Timing the training time
    # Initializing the optimizer for the model parameters
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=epochs, eta_min=learning_rate / 5
    )

    # Decay rate loss
    loss_function = nn.L1Loss()
    if aleatoric_uncertainty:
        beta = 0.25
        loss_function = NLL_loss

    ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    alpha_ssim = 0.1  # weight for SSIM loss

    # Region types loss
    ce_loss = nn.CrossEntropyLoss()
    weight_ce_loss = 0.0
    focal_loss = FocalLoss(
        to_onehot_y=True,
    )

    weight_focal_loss = 1.0 - weight_ce_loss

    best_val_loss = np.inf
    wait = 0  # for early stopping
    training_losses = []
    val_losses = []
    with open(losses_dir[:-4] + "-running.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training Loss", "Validation Loss"])
    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0

        # Training loop
        batch = 0
        for batch_input, batch_target, _, _ in tqdm(train_loader):
            loss = 0
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            batch_output = model(batch_input)  # generating images

            if region_types_flag:
                loss += weight_ce_loss * ce_loss(
                    batch_output, batch_target.squeeze(1)
                ) + weight_focal_loss * focal_loss(batch_output, batch_target)
            elif aleatoric_uncertainty:
                batch_output_mean = batch_output[:, 0].unsqueeze(1)
                batch_output_var = batch_output[:, 1].unsqueeze(1)
                loss += (1 - alpha_ssim) * loss_function(
                    batch_output_mean, batch_output_var, batch_target, beta=beta
                ) + alpha_ssim * (1 - ssim_loss(batch_output_mean, batch_target))
            else:
                loss += (1 - alpha_ssim) * loss_function(
                    batch_output, batch_target
                ) + alpha_ssim * (1 - ssim_loss(batch_output, batch_target))
            loss.backward()  # backprop

            if batch % accumulation_steps == 0:
                optim.step()
                optim.zero_grad()  # resetting gradients

            train_loss += loss.item()
            batch += 1
            with open(timing_dir, "w") as file:
                file.write(f"epoch {epoch} batch {batch}\n")

        # Validation loop
        with torch.no_grad():
            for batch_input, batch_target, _, _ in tqdm(val_loader):
                loss = 0
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                batch_output = model(batch_input)  # generating images
                if region_types_flag:
                    loss += weight_ce_loss * ce_loss(
                        batch_output, batch_target.squeeze(1)
                    ) + weight_focal_loss * focal_loss(batch_output, batch_target)
                elif aleatoric_uncertainty:
                    batch_output_mean = batch_output[:, 0].unsqueeze(1)
                    batch_output_var = batch_output[:, 1].unsqueeze(1)
                    # beta (regularization term for NLL loss) is set to 0 for validation because the total loss may increase, since the regularization term is detached, but we are interested in saving the best loss without regularization
                    loss += (1 - alpha_ssim) * loss_function(
                        batch_output_mean, batch_output_var, batch_target, beta=0.0
                    ) + alpha_ssim * (1 - ssim_loss(batch_output_mean, batch_target))
                else:
                    loss += (1 - alpha_ssim) * loss_function(
                        batch_output, batch_target
                    ) + alpha_ssim * (1 - ssim_loss(batch_output, batch_target))
                val_loss += loss.item()

        scheduler.step()

        # Calculate average losses (to make it independent of batch size)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch} training loss: {avg_train_loss}")
        print(f"Epoch {epoch} validation loss: {avg_val_loss}")

        # Log the losses for plotting
        training_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        save_plot_file = os.path.join(save_plot_dir, "loss-running.jpg")
        plot_losses(training_losses, val_losses, save_plot_file)
        # Save losses
        with open(losses_dir[:-4] + "-running.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, avg_train_loss, avg_val_loss])

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            torch.save(model, model_dir)
        else:
            wait += 1
            if wait >= patience:
                print(f"Stopping early at epoch {epoch}")
                epoch = epoch - patience
                break

    # End and save timing, plots and definitive losses
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time} seconds")
    # Save to file
    with open(timing_dir, "w") as file:
        file.write(f"Training time: {elapsed_time} seconds. Best epoch: {epoch}")

    os.rename(save_plot_file, os.path.join(save_plot_dir, "loss.jpg"))
    os.rename(losses_dir[:-4] + "-running.csv", losses_dir)

    return model
