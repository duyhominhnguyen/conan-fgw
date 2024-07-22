from typing import Optional, Callable

import torch
from torch import Tensor, LongTensor
from torch.nn import ModuleList
from torch_geometric.nn import SchNet, aggr
from torch_geometric.nn.models.schnet import InteractionBlock
from torch_geometric.typing import OptTensor
from torch_geometric.utils import to_dense_adj, to_dense_batch

from conan_fgw.src.model.fgw.barycenter import fgw_barycenters, normalize_tensor
from tqdm import tqdm
import torch.nn.functional as F
import random
import pickle
import os
import numpy as np
import time
import uuid
import pandas as pd
import shutil

COVALENT_BONDS_ATTRS_DIM = 3


def get_list_node_features(out, mask):
    """
    Convert outputs of to_dense_batch to a list of different feature node matrices
    :param out:
    :param mask:
    :return:
    """
    our_filter = []
    for index, sample in enumerate(out):
        mask_index = mask[index]
        pos_true = (mask_index == True).nonzero().squeeze()
        our_filter.append(sample[pos_true])
    return our_filter


def get_list_node_features_batch(batch_size, num_conformers, out_ft_dense, mask_ft_dense):
    """
    - Given node features and masks output after calling out_ft_dense, mask_ft_dense = to_dense_batch(x, batch),
    return 'out_total' which is a list of node features for each batch size.
    len(out_total) is a batch size, len(out_total[i]) is the number of conformer graphs of each molecular;
    out_total[i][j] is the node feature matrices of the j-th conformer.
    Args:
        batch_size:
        num_conformers:
        out_ft_batch:
        mask_ft_batch:

    Returns: out_total

    """
    out_total = []
    start_index = 0
    for index in range(int(batch_size)):
        out_ft_batch = out_ft_dense[start_index : start_index + num_conformers, :, :] + 0.5
        mask_ft_batch = mask_ft_dense[start_index : start_index + num_conformers, :]
        start_index = start_index + num_conformers

        # remove redundancy
        # out_ft_filter = get_list_node_features(out_ft_batch, mask_ft_batch)
        # out_total.append(out_ft_filter)
        out_total.append([normalize_tensor(item, 0.1, 2.0) for item in out_ft_batch])
    return out_total


def get_adj_dense_batch(batch_size, num_conformers, adj_dense):
    """
    Given adj = to_dense_adj(edge_index=edge_index, batch=batch), return
    a list of adjacency matrices for each sample in batch size.
    Args:
        batch_size:
        num_conformers:
        adj_dense:

    Returns:

    """
    start_index = 0
    adj_dense_batch = []
    for index in range(int(batch_size)):
        adj_dense_batch.append(adj_dense[start_index : start_index + num_conformers, :, :])
        start_index = start_index + num_conformers
    return adj_dense_batch


class SchNetNoSum(SchNet):
    def __init__(
        self,
        device,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = "add",
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: OptTensor = None,
        use_covalent: bool = False,
        use_readout: bool = True,
    ):
        super().__init__(
            hidden_channels,
            num_filters,
            num_interactions,
            num_gaussians,
            cutoff,
            interaction_graph,
            max_num_neighbors,
            readout,
            dipole,
            mean,
            std,
            atomref,
        )
        self.device = device
        self.use_readout = use_readout
        self.use_covalent = use_covalent
        self.lin1_bary = torch.nn.Linear(
            hidden_channels, hidden_channels // 2
        )  # MLP for barycenter
        self.lin2_bary = torch.nn.Linear(hidden_channels // 2, hidden_channels // 2)
        self.lin2 = torch.nn.Linear(hidden_channels // 2, hidden_channels // 2)

        if self.use_covalent:
            self.interactions_cov = ModuleList()
            for _ in range(num_interactions):
                block = InteractionBlock(
                    hidden_channels, COVALENT_BONDS_ATTRS_DIM, num_filters, cutoff
                )
                self.interactions_cov.append(block)
            self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels // 2)
            self.lin1_bary = torch.nn.Linear(
                hidden_channels * 2, hidden_channels // 2
            )  # MLP for barycenter

    def forward(self, z: Tensor, pos: Tensor, batch: OptTensor = None, data_batch=None) -> Tensor:
        r"""
        Args:
            z (LongTensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
            data_batch (Batch, optional): Batch object with additional data, such as covalent bonds attributes
        """

        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        if self.use_covalent:
            h_cov = self.embedding(z)
            edge_index_cov = data_batch.edge_index
            edge_weight_cov = torch.ones(edge_index_cov.shape[1], dtype=torch.float32).to(
                self.device
            )
            edge_attr_cov = data_batch.edge_attr.float()
            for interaction in self.interactions_cov:
                h_cov = h_cov + interaction(h_cov, edge_index_cov, edge_weight_cov, edge_attr_cov)
            h = torch.cat([h, h_cov], dim=1)

        h = self.lin1(h)
        h = self.lin2(h)
        h = self.act(h)
        # We don't need this layer because we want to aggregate embeddings not scalar values
        # h = self.lin2(h)

        if self.use_readout:
            out = self.readout(h, batch, dim=0)
        else:
            out = h

        return out

    def forward_3d_bary(
        self, z: Tensor, pos: Tensor, batch: OptTensor = None, data_batch=None
    ) -> Tensor:
        """
        Create two embeddings, one for standard 3D aggregation, second for barycenter
            Args:
        z (LongTensor): Atomic number of each atom with shape
            :obj:`[num_atoms]`.
        pos (Tensor): Coordinates of each atom with shape
            :obj:`[num_atoms, 3]`.
        batch (LongTensor, optional): Batch indices assigning each atom to
            a separate molecule with shape :obj:`[num_atoms]`.
            (default: :obj:`None`)
        data_batch (Batch, optional): Batch object with additional data, such as covalent bonds attributes
        """
        batch = torch.zeros_like(z) if batch is None else batch

        h_shared = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h_shared = h_shared + interaction(h_shared, edge_index, edge_weight, edge_attr)

        if self.use_covalent:
            h_cov = self.embedding(z)
            edge_index_cov = data_batch.edge_index
            edge_weight_cov = torch.ones(edge_index_cov.shape[1], dtype=torch.float32).to(
                self.device
            )
            edge_attr_cov = data_batch.edge_attr.float()
            for interaction in self.interactions_cov:
                h_cov = h_cov + interaction(h_cov, edge_index_cov, edge_weight_cov, edge_attr_cov)
            h_shared = torch.cat([h_shared, h_cov], dim=1)

        h = self.lin1(h_shared)
        h = self.lin2(h)
        h = self.act(h)

        h_bary = self.lin1_bary(h_shared)
        h_bary = self.lin2_bary(h_bary)
        h_bary = self.act(h_bary)
        return h, h_bary

    def _compute_barycenter(
        self,
        node_feature: torch.Tensor,  ##  Node feature matrix [h, pos] if else outter
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
        num_conformers: int,
    ):
        out_ft_dense, mask_ft_dense = to_dense_batch(
            x=node_feature, batch=batch  ## [Nc*B*nodes] -> sum(nodes)
        )
        out_ft_batch = get_list_node_features_batch(
            batch_size, num_conformers, out_ft_dense, mask_ft_dense
        )

        adj_dense = to_dense_adj(
            edge_index=edge_index,
            batch=batch,
        )
        adj_dense_batch = get_adj_dense_batch(batch_size, num_conformers, adj_dense)
        F_bary_batch = torch.zeros(
            (batch_size * num_conformers, node_feature.shape[1]), device=self.device
        )
        index_bary = 0

        for index in range(batch_size):

            out_ft_sample = out_ft_batch[index]
            adj_dense_sample = adj_dense_batch[index]
            list_adjs = [item for item in adj_dense_sample]
            if adj_dense_sample[0].get_device() >= 0:
                w_tmp = [
                    torch.ones(t.shape[0], dtype=torch.float32).to(
                        adj_dense_sample[0].get_device()
                    )
                    / t.shape[0]
                    for t in list_adjs
                ]
                lambdas = torch.ones(len(list_adjs), dtype=torch.float32).to(
                    adj_dense_sample[0].get_device()
                ) / len(list_adjs)
            else:
                w_tmp = [
                    torch.ones(t.shape[0], dtype=torch.float32) / t.shape[0] for t in list_adjs
                ]
                lambdas = torch.ones(len(list_adjs), dtype=torch.float32) / len(list_adjs)

            F_bary, C_bary, log = fgw_barycenters(
                N=adj_dense_sample.shape[1],
                Ys=out_ft_sample,
                Cs=list_adjs,
                ps=w_tmp,
                lambdas=lambdas,
                warmstartT=True,
                symmetric=True,
                method="sinkhorn_log",
                alpha=0.1,
                solver="PGD",
                fixed_structure=False,
                fixed_features=False,
                epsilon=0.1,
                p=None,
                loss_fun="square_loss",
                max_iter=5,
                tol=1e-2,
                numItermax=5,
                stopThr=1e-2,
                verbose=False,
                log=True,
                init_C=list_adjs[0],
                init_X=None,
                random_state=None,
            )

            h_out_bary = self.readout(F_bary, dim=0)
            F_bary_batch[index_bary : index_bary + num_conformers, :] = h_out_bary.repeat(
                num_conformers, 1
            )
            index_bary = index_bary + num_conformers

        node_feature = self.readout(node_feature, batch, dim=0)
        return (node_feature, F_bary_batch)

    def forward_w_barycenter(
        self,
        z: Tensor,
        pos: Tensor,
        num_conformers: int,
        batch: OptTensor = None,
        data_batch=None,
        max_iter: int = 100,
        epsilon: float = 0.1,
    ) -> Tensor:
        r"""
        Args:
            z (LongTensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
            data_batch (Batch, optional): Batch object with additional data, such as covalent bonds attributes
        """
        batch = torch.zeros_like(z) if batch is None else batch

        # h = self.embedding(z)
        h_3d, h_bary = self.forward_3d_bary(z, pos, batch)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        # edge_attr = self.distance_expansion(edge_weight)

        batch_size = int(len(batch.unique()) / num_conformers)
        h_3d_non, h_bary = self._compute_barycenter(
            node_feature=h_bary,
            edge_index=edge_index,
            batch=batch,
            batch_size=batch_size,
            num_conformers=num_conformers,
        )
        h_3d = self.readout(h_3d, batch, dim=0)
        return h_3d, h_bary


class SchNetWithMultipleReturns(SchNet):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = "add",
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: OptTensor = None,
        use_covalent: bool = False,
        use_readout: bool = True,
    ):
        super().__init__(
            hidden_channels,
            num_filters,
            num_interactions,
            num_gaussians,
            cutoff,
            interaction_graph,
            max_num_neighbors,
            readout,
            dipole,
            mean,
            std,
            atomref,
        )
        self.use_readout = use_readout
        self.use_covalent = use_covalent
        if self.use_covalent:
            self.interactions_cov = ModuleList()
            for _ in range(num_interactions):
                block = InteractionBlock(
                    hidden_channels, COVALENT_BONDS_ATTRS_DIM, num_filters, cutoff
                )
                self.interactions_cov.append(block)
            self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels // 2)

            # self.covalent_linear = torch.nn.Linear(COVALENT_BONDS_ATTRS_DIM, hidden_channels)
            # self.assembled_linear = torch.nn.Linear(hidden_channels, hidden_channels)
        self.mean_aggregation = aggr.MeanAggregation()
        self.sum_aggregation = aggr.SumAggregation()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: OptTensor = None,
        data_batch=None,
        conformers_index=None,
    ):
        r"""
        Args:
            z (LongTensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
            data_batch (Batch, optional): Batch object with additional data, such as covalent bonds attributes
        """
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)

        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        if self.use_covalent:
            h_cov = self.embedding(z)
            edge_index_cov = data_batch.edge_index
            edge_weight_cov = torch.ones(edge_index_cov.shape[1], dtype=torch.float32).to(
                self.device
            )
            edge_attr_cov = data_batch.edge_attr.float()
            for interaction in self.interactions_cov:
                h_cov = h_cov + interaction(h_cov, edge_index_cov, edge_weight_cov, edge_attr_cov)
            h = torch.cat([h, h_cov], dim=1)

        h = self.lin1(h)
        h = self.act(h)
        # We don't need this layer because we want to aggregate embeddings not scalar values
        # h = self.lin2(h)

        return h, edge_index, edge_attr
