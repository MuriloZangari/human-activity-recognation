"""
data_loader.py

Contains the PyTorch Dataset class used to wrap the Human Activity Recognition dataset.
This dataset provides feature vectors (561 dimensions) and corresponding activity labels (encoded as integers).

Author: Murilo Zangari
"""

import torch
from torch.utils.data import Dataset

class HumanActivityDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (np.ndarray or torch.Tensor): Feature matrix with shape [n_samples, 561]
            labels (np.ndarray or torch.Tensor): Label vector with shape [n_samples]
        """
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)  # required for CrossEntropyLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
