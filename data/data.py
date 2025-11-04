# data.py
from typing import Optional
import os
import torch
import pandas as pd
import numpy as np

# ✅ Only one dataset — your acetone dataset
DATASETS = ['batch_dist_ternary_acetone_1_butanol_methanol']


class NoBoomDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: str, version: str = '1.0', root: str = '.', train: bool = True,
                 binary_labels: bool = False, download: bool = False, train_anomalies: bool = False,
                 include_misc_faults: bool = False, include_controller_faults: bool = False):

        super().__init__()

        assert dataset in DATASETS, f"Dataset must be one of {DATASETS}"

        self.dataset = dataset
        self.version = version
        self.root = root
        self.train = train
        self.binary_labels = binary_labels
        self.train_anomalies = train_anomalies
        self.include_misc_faults = include_misc_faults
        self.include_controller_faults = include_controller_faults

        self.data: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self._features: list[str] = []

        if download:
            self.download()

        self.load(self.filter_file_list(self.file_list()))

    def download(self):
        raise NotImplementedError()

    def load(self, file_list: list[str]):
        """Loads the data from disk into memory."""
        label_feature, label_misc_feature, label_controller_feature, meta_data = self.meta_data()

        for file in file_list:
            time_series = pd.read_csv(os.path.join(self.root, file))

            # Merge soft/controller faults if requested
            if self.include_misc_faults and label_misc_feature is not None:
                mask = (time_series[label_misc_feature] > 0) & (time_series[label_feature] == 0)
                time_series.loc[mask, label_feature] = time_series.loc[mask, label_misc_feature]

            if self.include_controller_faults and label_controller_feature is not None:
                mask = (time_series[label_controller_feature] > 0) & (time_series[label_feature] == 0)
                time_series.loc[mask, label_feature] = time_series.loc[mask, label_controller_feature]

            # Store labels
            if self.binary_labels:
                self.targets.append(time_series[label_feature].to_numpy() / time_series[label_feature].to_numpy())
            else:
                self.targets.append(time_series[label_feature].to_numpy())

            # Drop non-feature columns
            time_series.drop(meta_data, axis=1, inplace=True)
            time_series = time_series.astype(np.float32)
            self.data.append(time_series.to_numpy())
            self._features = list(time_series)

            del time_series

    def file_list(self) -> list[str]:
        with open(os.path.join('..', 'metadata', 'datasets', f'{self.dataset}_{self.version}.txt')) as file:
            return file.read().split('\n')

    def filter_file_list(self, file_list: list[str]) -> list[str]:
        """
        Filter which files belong to train/test split.
        Special rule: all CSVs inside 'operating_point_001' are always
        considered training data, even if they start with 'test_'.
        """
        prefixes = []
        if self.train:
            prefixes.append('train_normal')
        else:
            prefixes.extend(['test_normal', 'test_anormal'])

        if self.train_anomalies:
            prefixes.append('train_anormal')

        selected = []
        for file in file_list:
            # ✅ Override rule for this dataset
            if 'operating_point_001' in file and self.train:
                selected.append(file)
                continue
            # Normal filtering logic
            if any(pref in file for pref in prefixes):
                selected.append(file)
        return selected

    def __getitem__(self, item: int) -> tuple[tuple[torch.Tensor], tuple[torch.Tensor]]:
        return (torch.as_tensor(self.data[item]),), (torch.as_tensor(self.targets[item]),)

    def __len__(self) -> Optional[int]:
        return len(self.data)

    @property
    def seq_len(self) -> list[int]:
        return [ts.data.shape[0] for ts in self.data]

    @property
    def num_features(self) -> int:
        return self.data[0].shape[-1]

    @property
    def features(self) -> list[str]:
        return self._features

    @staticmethod
    def meta_data() -> tuple[str, Optional[str], Optional[str], list[str]]:
        return (
            'Label (advanced/hard fault)',
            'Label (advanced/soft fault)',
            'Label (advanced/controller fault)',
            [
                'Time',
                'Label (common/hard fault)',
                'Label (common/soft fault)',
                'Label (common/controller fault)',
                'Label (common/hard and soft)',
                'Label (common/all)',
                'Label (advanced/hard fault)',
                'Label (advanced/soft fault)',
                'Label (advanced/controller fault)'
            ]
        )
