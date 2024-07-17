from typing import Optional

import numpy as np
import torch
import torch_geometric
from torch import Tensor
from torch_geometric.nn import aggr
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor

from conan_fgw.src.model.graph_embeddings.gat import GATBased
from conan_fgw.src.model.graph_embeddings.schnet_no_sum import (
    SchNetNoSum,
    SchNetWithMultipleReturns,
)


def get_per_positions_index(batch, conformers_index, device):
    per_position_index = [
        [] for _ in range(batch.max().item() + 1)
    ]  # list of listst: inner list represents atoms of a molecule
    per_conformer_index = [
        [] for _ in range(conformers_index.max() + 1)
    ]  # list of lists: inner list represents atoms of a conformer
    pos_counter = 0
    last_mol_index = conformers_index[0]
    for atom_idx, conformer_idx in enumerate(batch.cpu().numpy()):
        mol_idx = conformers_index[conformer_idx]
        positions = per_position_index[conformer_idx]

        if mol_idx != last_mol_index:  # start new molecule
            last_mol_index = mol_idx
        elif len(positions) == 0 and pos_counter != 0:
            pos_counter -= len(per_position_index[batch[atom_idx - 1]])
            if len(per_conformer_index[mol_idx]) == 0:
                per_conformer_index[mol_idx] = [
                    mol_idx for _ in range(len(per_position_index[batch[atom_idx - 1]]))
                ]

        positions.append(pos_counter)
        per_position_index[conformer_idx] = positions
        pos_counter += 1

    per_position_index = [item for sublist in per_position_index for item in sublist]
    per_conformer_index = [item for sublist in per_conformer_index for item in sublist]
    return torch.LongTensor(per_position_index).to(device), torch.stack(
        per_conformer_index
    ).long().to(device)


def get_subgraph(batch, edge_index):
    edge_index_arr = edge_index.cpu().numpy()
    new_edge_index = []
    for edge in batch.edge_index.t():
        if np.isin(edge.cpu().numpy(), edge_index_arr).all():
            new_edge_index.append(edge)
    new_edge_index = torch.stack(new_edge_index, dim=0).t()
    return new_edge_index


class DeepSetsAggregation(Aggregation):
    r"""Performs Deep Sets aggregation in which the elements to aggregate are
    first transformed by a Multi-Layer Perceptron (MLP)
    :math:`\phi_{\mathbf{\Theta}}`, summed, and then transformed by another MLP
    :math:`\rho_{\mathbf{\Theta}}`, as suggested in the `"Graph Neural Networks
    with Adaptive Readouts" <https://arxiv.org/abs/2211.04952>`_ paper.

    Args:
        local_nn (torch.nn.Module, optional): The neural network
            :math:`\phi_{\mathbf{\Theta}}`, *e.g.*, defined by
            :class:`torch.nn.Sequential` or
            :class:`torch_geometric.nn.models.MLP`. (default: :obj:`None`)
        global_nn (torch.nn.Module, optional): The neural network
            :math:`\rho_{\mathbf{\Theta}}`, *e.g.*, defined by
            :class:`torch.nn.Sequential` or
            :class:`torch_geometric.nn.models.MLP`. (default: :obj:`None`)
    """

    def __init__(
        self,
        local_nn: Optional[torch.nn.Module] = None,
        global_nn: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.local_nn = local_nn
        self.global_nn = global_nn

    def reset_parameters(self):
        if self.local_nn is not None:
            reset(self.local_nn)
        if self.global_nn is not None:
            reset(self.global_nn)

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        if self.local_nn is not None:
            x = self.local_nn(x)
        x = self.reduce(x, index, ptr, dim_size, dim, reduce="sum")
        if self.global_nn is not None:
            x = self.global_nn(x)
        return x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(local_nn={self.local_nn}, " f"global_nn={self.global_nn})"
        )


class AverageConformerESAN(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.siamese = SchNetNoSum(use_covalent=False, use_readout=False)
        self.info_sharing = SchNetNoSum(use_covalent=False, use_readout=False)
        self.z_mean_aggregation = aggr.MeanAggregation()
        self.mean_aggregation = aggr.MeanAggregation()
        self.sum_aggregation = aggr.SumAggregation()
        self.hidden_channels = self.siamese.hidden_channels // 2
        self.deep_sets_aggregation = DeepSetsAggregation(
            local_nn=torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        )

    def forward(
        self, z: Tensor, pos: Tensor, batch: OptTensor, data_batch, conformers_index
    ) -> Tensor:
        per_position_index, per_conformer_index = get_per_positions_index(
            batch, conformers_index, self.device
        )

        return self.h_layer(
            z, pos, batch, data_batch, per_position_index, per_conformer_index, conformers_index
        )

    def h_layer(
        self,
        z: Tensor,
        pos: Tensor,
        batch: OptTensor,
        data_batch,
        per_position_index,
        per_conformer_index,
        conformers_index,
    ):
        # Let's make average conformer from averaging z positions
        z_avg = self.mean_aggregation(z.unsqueeze(dim=1), per_position_index).long().view(-1)
        pos_avg = self.mean_aggregation(pos, per_position_index)
        h_shared = self.info_sharing(z_avg, pos_avg, per_conformer_index)
        h_shared = self.sum_aggregation(h_shared, per_conformer_index)

        # Let's sum two embeddings
        # The siamese part returns per conformer embeddings
        h = self.siamese(z, pos, batch, data_batch)
        h = self.sum_aggregation(h, batch)
        h = self.deep_sets_aggregation(h, conformers_index)

        h = h + h_shared

        return h


class GeometryInducedESAN(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.siamese = SchNetWithMultipleReturns()
        self.gat_2D = GATBased()
        self.gat = GATBased(edge_dim=50)
        self.info_sharing = SchNetNoSum(use_covalent=False, use_readout=False)
        self.z_mean_aggregation = aggr.MeanAggregation()
        self.mean_aggregation = aggr.MeanAggregation()
        self.sum_aggregation = aggr.SumAggregation()
        self.hidden_channels = self.siamese.hidden_channels // 2
        self.deep_sets_aggregation = DeepSetsAggregation(
            local_nn=torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        )

        self.transformation_matrix = torch.nn.Linear(self.hidden_channels, self.hidden_channels)

    def forward(
        self, z: Tensor, pos: Tensor, batch: OptTensor, data_batch, conformers_index
    ) -> Tensor:
        per_position_index, per_conformer_index = get_per_positions_index(
            batch, conformers_index, self.device
        )

        return self.h_layer(
            z, pos, batch, data_batch, per_position_index, per_conformer_index, conformers_index
        )

    def h_layer(
        self,
        z: Tensor,
        pos: Tensor,
        batch: OptTensor,
        data_batch,
        per_position_index,
        per_conformer_index,
        conformers_index,
    ):
        # The siamese part returns per conformer embeddings
        h_3d, edge_index_3d, edge_attr_3d = self.siamese(z, pos, batch, data_batch)
        # new_edge_index = get_subgraph(data_batch, edge_index_3d)
        h_3d = self.sum_aggregation(h_3d, batch)

        # Let's use 2D subgraphs
        h_2d = self.h_layer_2d(data_batch, edge_index_3d, edge_attr_3d)
        h_2d = self.transformation_matrix(h_2d)
        h = h_3d + h_2d

        h = self.deep_sets_aggregation(h, conformers_index)

        # Let's make average conformer from averaging z positions
        z_avg = self.mean_aggregation(z.unsqueeze(dim=1), per_position_index).long().view(-1)
        pos_avg = self.mean_aggregation(pos, per_position_index)
        h_shared = self.info_sharing(z_avg, pos_avg, per_conformer_index)
        h_shared = self.sum_aggregation(h_shared, per_conformer_index)

        h = h + h_shared

        return h

    def h_layer_2d(self, data_batch, edge_index_3d, edge_attr_3d):
        # Let's use 2D subgraphs
        x_2d = self.gat_2D(
            data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch
        )
        subgraphs_x_2d = self.gat(data_batch.x, edge_index_3d, edge_attr_3d, data_batch.batch)
        return x_2d + subgraphs_x_2d


class Geometry2DInducedESAN(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.siamese = SchNetWithMultipleReturns()
        self.gat_2D = GATBased()
        self.gat_subgraphs = GATBased(edge_dim=3)
        self.info_sharing = SchNetNoSum(use_covalent=False, use_readout=False)
        self.z_mean_aggregation = aggr.MeanAggregation()
        self.mean_aggregation = aggr.MeanAggregation()
        self.sum_aggregation = aggr.SumAggregation()
        self.hidden_channels = self.siamese.hidden_channels // 2
        self.deep_sets_aggregation = DeepSetsAggregation(
            local_nn=torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        )

        self.transformation_matrix = torch.nn.Linear(self.hidden_channels, self.hidden_channels)

    def forward(
        self, z: Tensor, pos: Tensor, batch: OptTensor, data_batch, conformers_index
    ) -> Tensor:
        per_position_index, per_conformer_index = get_per_positions_index(
            batch, conformers_index, self.device
        )

        return self.h_layer(
            z, pos, batch, data_batch, per_position_index, per_conformer_index, conformers_index
        )

    def h_layer(
        self,
        z: Tensor,
        pos: Tensor,
        batch: OptTensor,
        data_batch,
        per_position_index,
        per_conformer_index,
        conformers_index,
    ):
        # The siamese part returns per conformer embeddings
        h_3d, edge_index_3d, _ = self.siamese(z, pos, batch, data_batch)
        # new_edge_index = get_subgraph(data_batch, edge_index_3d)

        adj = torch_geometric.utils.to_dense_adj(
            edge_index=data_batch.edge_index,
            edge_attr=data_batch.edge_attr,
            max_num_nodes=z.shape[0],
        ).squeeze(0)
        subgraph_edge_attr = adj[edge_index_3d[0], edge_index_3d[1]]

        # Let's use 2D subgraphs
        h_2d = self.h_layer_2d(data_batch, edge_index_3d, subgraph_edge_attr)
        h = self.deep_sets_aggregation(h_2d, conformers_index)

        h = self.transformation_matrix(h)

        # Let's make average conformer from averaging z positions
        z_avg = self.mean_aggregation(z.unsqueeze(dim=1), per_position_index).long().view(-1)
        pos_avg = self.mean_aggregation(pos, per_position_index)
        h_shared = self.info_sharing(z_avg, pos_avg, per_conformer_index)
        h_shared = self.sum_aggregation(h_shared, per_conformer_index)

        h = h + h_shared

        return h

    def h_layer_2d(self, data_batch, subgraph_edge_index, subgraph_edge_attr):
        # Let's use 2D subgraphs
        subgraphs_x_2d = self.gat_subgraphs(
            data_batch.x, subgraph_edge_index, subgraph_edge_attr, data_batch.batch
        )
        x_2d = self.gat_2D(
            data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch
        )
        return x_2d + subgraphs_x_2d
