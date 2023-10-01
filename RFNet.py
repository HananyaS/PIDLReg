import os

import pandas as pd
import torch
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

from sklearn.metrics import roc_auc_score

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam


class RFNet(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        frac: float = 0.5,
        n_estimators: int = 100,
    ):
        super(RFNet, self).__init__()

        # create n_estimators random masks of size frac * input_dim with int values in [0, input_dim) without replacement

        self.masks = [
            torch.randperm(input_dim)[: int(input_dim * frac)]
            for _ in range(n_estimators)
        ]
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(input_dim * frac), hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(n_estimators)
            ]
        )

    def forward(self, x):
        # for each mask, randomly select the input samples and pass them through the corresponding MLP
        # inputs = torch.randint(
        #     low=0, high=x.shape[0], size=(len(self.masks), x.shape[0])
        # )
        # inputs = [x[i][:, mask] for i, mask in zip(inputs, self.masks)]
        inputs = [x[:, mask] for mask in self.masks]
        outputs = [mlp(input) for mlp, input in zip(self.mlps, inputs)]

        output = torch.stack(outputs).mean(dim=0)

        return output

    def predict(self, x):
        with torch.no_grad():
            return self(x)

    def score(self, x, y, metric="accuracy"):
        if isinstance(x, pd.DataFrame):
            x = torch.from_numpy(x.values)

        if isinstance(y, pd.DataFrame):
            y = torch.from_numpy(y.values)

        if metric == "accuracy":
            with torch.no_grad():
                return (self.predict(x).argmax(dim=1) == y).float().mean().item()

        elif metric.lower() == "auc":
            with torch.no_grad():
                return roc_auc_score(y, self.predict(x)[:, 1].numpy())

        else:
            raise NotImplementedError

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset_name: str,
        lr: float = 1e-2,
        n_epochs: int = 300,
        verbose: bool = True,
        early_stopping: int = 30,
        loss_type: str = "cross_entropy",
    ):
        early_stopping_flag = early_stopping > 0

        if early_stopping_flag:
            print(f"Early stopping is enabled with patience {early_stopping}.")
            min_val_loss = np.inf
            best_model = None
            epochs_without_improvement = 0

        optimizer = Adam(self.parameters(), lr=lr)
        min_val_loss = np.inf
        best_model = None
        epochs_without_improvement = 0

        train_acc, val_acc = [], []
        train_losses, val_losses = [], []

        if loss_type == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        elif loss_type == "mse":
            criterion = nn.MSELoss()

        for epoch in range(n_epochs):
            train_losses_per_epoch, val_losses_per_epoch = [], []

            for i, (X, y) in enumerate(train_loader):
                optimizer.zero_grad()

                output = self(X)

                if loss_type == "mse":
                    # convert y to one-hot
                    y = torch.zeros_like(output).scatter_(1, y.unsqueeze(1), 1)

                loss = criterion(output, y)

                train_losses_per_epoch.append(loss.item())

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                for i, (X, y) in enumerate(val_loader):
                    output = self(X)

                    if loss_type == "mse":
                        # convert y to one-hot
                        y = torch.zeros_like(output).scatter_(1, y.unsqueeze(1), 1)

                    loss = criterion(output, y)

                    val_losses_per_epoch.append(loss.item())

            if early_stopping_flag:
                if sum(val_losses_per_epoch) < min_val_loss:
                    min_val_loss = sum(val_losses_per_epoch)
                    best_model = deepcopy(self)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping:
                    print(f"Early stopping at epoch {epoch + 1}.")
                    self = best_model
                    break

            train_losses.append(sum(train_losses_per_epoch) / len(train_loader.dataset))
            val_losses.append(sum(val_losses_per_epoch) / len(val_loader.dataset))

            train_acc.append(
                self.score(
                    train_loader.dataset.X,
                    train_loader.dataset.y,
                    metric="accuracy",
                )
            )
            val_acc.append(
                self.score(
                    val_loader.dataset.X,
                    val_loader.dataset.y,
                    metric="accuracy",
                )
            )

            # if early_stopping_flag:
            #     if val_losses[-1] < min_val_loss:
            #         min_val_loss = val_losses[-1]
            #         best_model = deepcopy(self)
            #         epochs_without_improvement = 0
            #     else:
            #         epochs_without_improvement += 1
            #
            #     if epochs_without_improvement >= early_stopping:
            #         print(f"Early stopping at epoch {epoch + 1}.")
            #         self = best_model
            #         break

            if verbose:
                print(f"Epoch {epoch + 1}:")
                print(f"\tTrain Loss: {train_losses[-1]:.3f}")
                print(f"\tVal Loss: {val_losses[-1]:.3f}")
                print(f"\tTrain Acc: {train_acc[-1]:.3f}")
                print(f"\tTest Acc: {val_acc[-1]:.3f}")

        plt.clf()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.legend()
        plt.title(f"Losses for {dataset_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        os.makedirs("loss_plots", exist_ok=True)
        plt.savefig(f"loss_plots/{dataset_name}.png")

        plt.clf()
        plt.plot(train_acc, label="Train Acc")
        plt.plot(val_acc, label="Val Acc")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.title(f"Accuracies for {dataset_name}")

        os.makedirs("acc_plots", exist_ok=True)
        plt.savefig(f"acc_plots/{dataset_name}.png")

        return best_model
