import logging
from abc import ABCMeta, abstractmethod
import random
import pytorch_lightning as pl
import torch_geometric
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, Sampler

from torch_geometric.data import Dataset

from conan_fgw.src.data.datasets import LargeConformerBasedDataset

python_logger = logging.getLogger("datamodules")


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = labels[idx]
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = (
                len(self.dataset[label])
                if len(self.dataset[label]) > self.balanced_max
                else self.balanced_max
            )

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def __len__(self):
        return self.balanced_max * len(self.keys)


class BalancedDistributedBatchSampler(DistributedSampler):
    def __init__(self, dataset, labels=None):
        super().__init__(dataset=dataset)
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = labels[idx]
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = (
                len(self.dataset[label])
                if len(self.dataset[label]) > self.balanced_max
                else self.balanced_max
            )

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def __len__(self):
        return self.balanced_max * len(self.keys)


class MyDataModule(pl.LightningDataModule, metaclass=ABCMeta):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        is_distributed: bool,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.is_distributed = is_distributed

    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.get_dataloader(self.test_dataset)

    @staticmethod
    def get_distributed_sampler(dataset: Dataset):
        return DistributedSampler(dataset, shuffle=False, seed=42)

    @abstractmethod
    def get_dataloader(self, dataset: Dataset):
        pass


class SmilesBasedDataModule(MyDataModule):

    def get_dataloader(self, dataset):
        if self.is_distributed:
            return torch_geometric.loader.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                sampler=MyDataModule.get_distributed_sampler(dataset),
            )
        else:
            return torch_geometric.loader.DataLoader(dataset=dataset, batch_size=self.batch_size)


class ConformersBasedDataModule(MyDataModule):
    def get_dataloader(self, dataset):
        if self.is_distributed:
            return DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                sampler=MyDataModule.get_distributed_sampler(dataset),
                collate_fn=LargeConformerBasedDataset.collate_fn,
                num_workers=8,
            )
        else:
            return DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=LargeConformerBasedDataset.collate_fn,
                num_workers=8,
            )
