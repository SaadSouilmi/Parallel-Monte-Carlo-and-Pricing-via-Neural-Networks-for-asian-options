import torch
from torch.utils.data import Dataset
import numpy as np


class PayoffDataset(Dataset):
    """Custom dataset class containing sample parameters and corresponding sampled payoffs
    This custom dataset is used to directly learn the payoffs instead of and MC estimation of
    the price."""

    def __init__(self, sample_params, sample_payoffs):
        self.sample_params = torch.tensor(sample_params, dtype=torch.float)
        self.sample_payoffs = torch.tensor(sample_payoffs, dtype=torch.float)
        self.nb_samples = len(self.sample_params)
        self.nb_paths = self.sample_payoffs.shape[1]

    def __len__(self):
        return self.nb_samples * self.nb_paths

    def __getitem__(self, index):
        return (
            self.sample_params[index // self.nb_paths],
            self.sample_payoffs[index // self.nb_paths, index % self.nb_paths],
        )
