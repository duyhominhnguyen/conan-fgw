from typing import Optional, Callable, Tuple
import torch
from torch import Tensor, LongTensor
from torch.nn import ModuleList
from torch_geometric.nn import SchNet, aggr, radius_graph

# from torch_geometric.nn.models.visnet import ViSNet as NaiveViSNet
from conan_fgw.src.model.graph_embeddings.torch_geometric_visnet import ViSNet as NaiveViSNet
from conan_fgw.src.model.graph_embeddings.torch_geometric_visnet import ViSNetBlock
from torch_geometric.nn.models.schnet import InteractionBlock, GaussianSmearing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import to_dense_adj, to_dense_batch, scatter
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver

from torch_geometric.nn.models.schnet import RadiusInteractionGraph
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
from torch import linalg as LA
from torch.nn import Embedding, LayerNorm, Linear

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
        out_ft_batch = out_ft_dense[start_index : start_index + num_conformers, :, :] + 1.0
        mask_ft_batch = mask_ft_dense[start_index : start_index + num_conformers, :]
        start_index = start_index + num_conformers

        # remove redundancy
        # out_ft_filter = get_list_node_features(out_ft_batch, mask_ft_batch)
        # out_total.append(out_ft_filter)
        # out_total.append([normalize_tensor(item, 0.1, 2.) for item in out_ft_batch])
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


class ViSNet(NaiveViSNet):
    """
    lmax: int = 1,
    vecnorm_type: Optional[str] = None,
    trainable_vecnorm: bool = False,
    num_heads: int = 8,
    num_layers: int = 6,
    hidden_channels: int = 128,
    num_rbf: int = 32,
    trainable_rbf: bool = False,
    max_z: int = 100,
    cutoff: float = 5.0,
    max_num_neighbors: int = 32,
    vertex: bool = False,
    atomref: Optional[Tensor] = None,
    reduce_op: str = "sum",
    mean: float = 0.0,
    std: float = 1.0,
    derivative: bool = False,
    """

    def __init__(self, device, hidden_channels: int, cutoff: float = 5.0):
        super().__init__(
            hidden_channels=hidden_channels,
        )
        self.device = device
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.interaction_graph = RadiusInteractionGraph(cutoff, max_num_neighbors=32)
        self.readout = aggr_resolver("sum")

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Computes the energies or properties (forces) for a batch of
        molecules.

        Args:
            z (torch.Tensor): The atomic numbers.
            pos (torch.Tensor): The coordinates of the atoms.
            batch (torch.Tensor): A batch vector,
                which assigns each node to a specific example.

        Returns:
            y (torch.Tensor): The energies or properties for each molecule.
            dy (torch.Tensor, optional): The negative derivative of energies.
        """
        if self.derivative:
            pos.requires_grad_(True)

        x, v = self.representation_model(z, pos, batch)
        x = self.output_model.pre_reduce(x, v)
        x = x * self.std
        if self.prior_model is not None:
            x = self.prior_model(x, z)

        x = self.readout(x, batch, dim=0)
        return x

    def forward_3d_bary(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Computes the energies or properties (forces) for a batch of
        molecules.

        Args:
            z (torch.Tensor): The atomic numbers.
            pos (torch.Tensor): The coordinates of the atoms.
            batch (torch.Tensor): A batch vector,
                which assigns each node to a specific example.

        Returns:
            y (torch.Tensor): The energies or properties for each molecule.
            dy (torch.Tensor, optional): The negative derivative of energies.
        """
        # compute embedding used for 3D aggregating
        if self.derivative:
            pos.requires_grad_(True)

        x_shared, v_shared = self.representation_model(z, pos, batch)
        x = self.output_model.pre_reduce(x_shared, v_shared)
        x = x * self.std
        if self.prior_model is not None:
            x = self.prior_model(x, z)

        x_br = self.output_model_bary.pre_reduce(x_shared, v_shared)
        x_br = x_br * self.std
        if self.prior_model_bary is not None:
            x_br = self.prior_model_bary(x_br, z)

        return x, x_br

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

        adj_dense = to_dense_adj(edge_index=edge_index, batch=batch)
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

            ## check F_bary_batch
            F_bary_nan_idx = torch.isnan(F_bary)
            F_bary_nan = F_bary[F_bary_nan_idx]
            if len(F_bary_nan.flatten()) != 0:
                print(f"out_ft_sample {out_ft_sample}")
                print(f"🔴 F_bary_batch is NaN: {F_bary_nan}")
                F_bary = torch.zeros(F_bary.shape)

            norm_value = LA.norm(F_bary, dim=0)
            F_bary = F_bary / norm_value
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

        # h = self(z, pos, batch)     ## naive forward/passed onto interactions
        h_3d, h_bary = self.forward_3d_bary(z, pos, batch)
        edge_index, edge_weight = self.interaction_graph(pos, batch)

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
