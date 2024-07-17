import torch
import os
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.data import Dataset
from conan_fgw.src.data.datasets import LargeConformerBasedDataset
from torch.utils.data import DataLoader
import random
import logging


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logging.info(f"🐙 Seeding everything with: {seed}")


def load_dummy(model, dataset):
    dataloader = DataLoader(
        dataset=dataset, batch_size=64, collate_fn=LargeConformerBasedDataset.collate_fn
    )
    minibatch = next(iter(dataloader))
    # batch, node_index, conformers = minibatch
    batch, node_index = minibatch
    conformers_index = model.create_aggregation_index(batch)
    logging.info(f"🐳 Loading dummy into model for DDP running")
    model.forward_dummy(batch, conformers_index, node_index)
    return model
