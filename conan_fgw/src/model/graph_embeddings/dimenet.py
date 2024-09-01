from typing import Optional, Callable

import torch
from torch import Tensor, LongTensor
from torch.nn import ModuleList
from torch_geometric.nn import SchNet, aggr, radius_graph
from torch_geometric.nn import DimeNet as NaiveDimeNet
from torch_geometric.nn import DimeNetPlusPlus
from torch_geometric.nn.models.schnet import InteractionBlock, GaussianSmearing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import to_dense_adj, to_dense_batch, scatter
from torch_geometric.nn.models.dimenet import triplets

from conan_fgw.src.model.fgw.barycenter import fgw_barycenters, normalize_tensor, batch_fgw_barycenters_BAPG, Epsilon
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


class DimeNet(NaiveDimeNet):

    def __init__(
        self,
        device,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        num_bilinear: int = 8,
        num_spherical: int = 2,
        num_radial: int = 3,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        use_covalent: bool = False,  ## add-on
        num_interactions: int = 6,  ## add-on
        num_gaussians: int = 50,  ## add-on
        num_filters: int = 128,
    ):
        super().__init__(
            hidden_channels,
            out_channels,
            num_blocks,
            num_bilinear,
            num_spherical,
            num_radial,
            cutoff,
            envelope_exponent,
            num_before_skip,
            num_after_skip,
            num_output_layers,
        )
        self.device = device
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: OptTensor = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        if isinstance(self, DimeNetPlusPlus):
            pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
            a = (pos_ij * pos_jk).sum(dim=-1)
            b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        elif isinstance(self, DimeNet):
            pos_ji, pos_ki = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_i]
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.cross(pos_ji, pos_ki, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for kk, (interaction_block, output_block) in enumerate(
            zip(self.interaction_blocks, self.output_blocks[1:])
        ):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            sub_P = output_block(x, rbf, i, num_nodes=pos.size(0))
            P = P + sub_P

        P_scatter = scatter(P, batch, dim=0, reduce="sum")

        return P

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
            # edge_attr=edge_attr
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

            # F_bary, C_bary, log = fgw_barycenters(
            #     N=adj_dense_sample.shape[1],
            #     Ys=out_ft_sample,
            #     Cs=list_adjs,
            #     ps=w_tmp,
            #     lambdas=lambdas,
            #     warmstartT=True,
            #     symmetric=True,
            #     method="sinkhorn_log",
            #     alpha=0.5,
            #     solver="PGD",
            #     fixed_structure=True,
            #     fixed_features=False,
            #     epsilon=0.1,
            #     p=None,
            #     loss_fun="square_loss",
            #     max_iter=5,
            #     tol=1e-2,
            #     numItermax=5,
            #     stopThr=1e-2,
            #     verbose=False,
            #     log=True,
            #     init_C=list_adjs[0],
            #     init_X=None,
            #     random_state=None,
            # )

            out_ft_sample = torch.stack(out_ft_sample)
            list_adjs = torch.stack(list_adjs)
            w_tmp = torch.stack(w_tmp)
            rho = Epsilon(target=1., init=22, decay=0.5)
            F_bary, C_bary, log = batch_fgw_barycenters_BAPG(
                N=adj_dense_sample.shape[1],
                Ys=out_ft_sample,
                Cs=list_adjs,
                ps=w_tmp,
                lambdas=lambdas,
                alpha=0.5,
                fixed_structure=True,
                fixed_features=False,
                p=None,
                loss_fun="square_loss",
                max_iter=10,
                toly=1e-3, tolc=1e-3,
                rho=rho,
                verbose=False,
                log=True,
                init_C=list_adjs[0],
                init_X=None,
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

        h = self(z, pos, batch)  ## naive forward/passed onto interactions

        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )

        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)

        edge_attr = self.distance_expansion(edge_weight)
        batch_size = int(len(batch.unique()) / num_conformers)

        if "accuracy" in self.log:
            h_3d, h_bary = self._compute_barycenter(
                node_feature=h,
                edge_index=edge_index,
                batch=batch,
                batch_size=batch_size,
                num_conformers=num_conformers,
            )

        if "runtime" in self.log:
            minibatch_runtime = []
            start_time_base = time.time()
            h_3d, h_bary = self._compute_barycenter(
                node_feature=h,
                edge_index=edge_index,
                batch=batch,
                batch_size=batch_size,
                num_conformers=num_conformers,
            )
            end_time_base = time.time()
            minibatch_runtime.append(
                {"solver": self.solver, "runtime": end_time_base - start_time_base}
            )
            ## save each step
            step_id = str(uuid.uuid1()).split("-")[0]
            df_runtime = pd.DataFrame(minibatch_runtime)
            df_runtime.to_csv(os.path.join(self.logdir, f"{step_id}.csv"))

        else:
            h_3d, h_bary = self._compute_barycenter(
                node_feature=h,
                edge_index=edge_index,
                batch=batch,
                batch_size=batch_size,
                num_conformers=num_conformers,
            )
        return h_3d, h_bary
