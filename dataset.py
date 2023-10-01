import pandas as pd
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, shuffle: bool = True, norm: bool = True,
                 add_aug: bool = False):
        self.X = torch.from_numpy(X.values).to(device).float()
        self.y = torch.from_numpy(y.values).to(device)

        if shuffle:
            idx = torch.randperm(len(self.X))
            self.X = self.X[idx]
            self.y = self.y[idx]

        if norm:
            self.norm()

        self.aug_added = False

        if add_aug:
            self.add_augmentations()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def norm(self, mean: torch.Tensor = None, std: torch.Tensor = None, feats2norm: list = []):
        assert (mean is None or mean.shape[0] == self.X.shape[1]) and (
                std is None or std.shape[0] == self.X.shape[1]) and (
                       (mean is None) == (std is None))

        if mean is not None:
            self.X = (self.X - mean) / std
            return mean, std

        feats2norm_ = [i for i in range(self.X.shape[1]) if torch.unique(self.X[:, i]).shape[0] > 2]
        feats2norm = list(set(feats2norm_ + feats2norm))

        mean = torch.zeros(self.X.shape[1]).float().to(device)
        std = torch.ones(self.X.shape[1]).float().to(device)

        mean[feats2norm] = self.X[:, feats2norm].mean(dim=0).float()
        std[feats2norm] = self.X[:, feats2norm].std(dim=0).float()

        self.X = (self.X - mean) / std

        return mean, std

    def add_augmentations(self):
        if self.aug_added:
            return self.X, self.y

        mean = self.X.mean(dim=0)

        X_orig, y_orig = self.X.clone(), self.y.clone()

        for j in range(self.X.shape[1]):
            aug_X = X_orig.clone()
            aug_X[:, j] = mean[j]

            self.X = torch.cat([self.X, aug_X], dim=0)
            self.y = torch.cat([self.y, y_orig], dim=0)

        idx = torch.randperm(len(self.X))
        self.X = self.X[idx]
        self.y = self.y[idx]

        self.aug_added = True

        return self.X, self.y
