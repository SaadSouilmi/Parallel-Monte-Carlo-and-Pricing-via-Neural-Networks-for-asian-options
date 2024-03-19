from torch.utils.data import Dataset
import numpy as np


class PayoffDataset(Dataset):

    def __init__(self, sample_params, sample_payoffs):
        self.sample_params = sample_params
        self.sample_payoffs = sample_payoffs
        self.nb_samples = len(self.sample_params)
        self.nb_paths = self.sample_payoffs.shape[1]

    def __len__(self):
        return self.nb_samples * self.nb_paths

    def __getitem__(self, index):
        return (
            self.sample_params[index // self.nb_paths],
            self.sample_payoffs[index // self.nb_paths, index % self.nb_paths],
        )
