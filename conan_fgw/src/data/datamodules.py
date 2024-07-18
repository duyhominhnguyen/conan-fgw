import logging
from abc import ABCMeta, abstractmethod
import random
import pytorch_lightning as pl
import torch_geometric
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, Sampler

from torch_geometric.data import Dataset

from conan_fgw.src.data.datasets import LargeConformerBasedDataset


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
