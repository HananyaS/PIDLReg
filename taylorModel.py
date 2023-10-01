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


class TaylorModel(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_powers: int = 3,
    ):
        super(TaylorModel, self).__init__()
        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, output_dim, bias=i == 0) for i in range(n_powers)]
        )

    def forward(self, x):
        xs = [x ** (i + 1) for i in range(1, len(self.fcs) + 1)]
        output = sum([fc(x) for fc, x in zip(self.fcs, [x] + xs)])

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
    ):
        optimizer = Adam(self.parameters(), lr=lr)
        min_val_loss = np.inf
        best_model = None
        epochs_without_improvement = 0

        train_acc, val_acc = [], []
        train_losses, val_losses = [], []

        criterion = nn.CrossEntropyLoss()

        for epoch in range(n_epochs):
            train_losses_per_epoch, val_losses_per_epoch = [], []

            for i, (X, y) in enumerate(train_loader):
                optimizer.zero_grad()

                output = self(X)
                loss = criterion(output, y)

                train_losses_per_epoch.append(loss.item())

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                for i, (X, y) in enumerate(val_loader):
                    output = self(X)
                    loss = criterion(output, y)

                    val_losses_per_epoch.append(loss.item())

            if sum(val_losses_per_epoch) < min_val_loss:
                min_val_loss = sum(val_losses_per_epoch)
                best_model = deepcopy(self)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

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
