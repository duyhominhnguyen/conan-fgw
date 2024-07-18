import torch
from torch.nn import Linear, Softmax, ReLU, Dropout, BatchNorm1d, Sequential
from torch_geometric.nn import aggr
import torch.nn.functional as F
from torchmetrics.functional import auroc

from conan_fgw.src.model.common import (
    EquivAggregation,
    EquivModelsHolder,
    EquivAggregationClassification,
)
from conan_fgw.src.model.attention_layer import Attention_Layer, SelfAttention
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
import os


def build_mlp(out_channels: int, is_complex: bool = False):
    if is_complex:
        mlp = Sequential(
            Linear(out_channels, out_channels // 2),
            Dropout(0.02),
            ReLU(),
            Linear(out_channels // 2, 1),
            Dropout(0.02),
        )
    else:
        mlp = Linear(out_channels, 1)
    return mlp


def build_mlp_class(out_channels: int, is_complex: bool = False):
    if is_complex:
        mlp = Sequential(
            Linear(out_channels, out_channels),
            # Dropout(0.02),
            ReLU(),
            Linear(out_channels, out_channels // 2),
            ReLU(),
            # BatchNorm1d(out_channels//2),
            Linear(out_channels // 2, 1),
            # Dropout(0.02),
        )
    else:
        mlp = Linear(out_channels, 1)
    return mlp


class ScalarsAggregation(EquivAggregation):
    def __init__(
        self,
        num_conformers: int,
        batch_size: int,
        learning_rate: float,
        model_name="simple_schnet",
        is_distributed: bool = False,
    ):
        super().__init__(num_conformers, batch_size, learning_rate, model_name, is_distributed)

    def forward(self, batch, conformers_index, node_index):
        x: torch.Tensor = self.node_embeddings_model(batch.z, batch.pos, node_index)
        x = self.conformers_mean_aggr(x, conformers_index)
        return x


class EmbeddingsAggregation(EquivAggregation):
    def __init__(
        self,
        num_conformers: int,
        batch_size: int,
        learning_rate: float,
        model_name="schnet",
        is_distributed: bool = False,
    ):
        super().__init__(num_conformers, batch_size, learning_rate, model_name, is_distributed)
        self.molecular_regression_lin = Linear(self.node_embeddings_model.hidden_channels // 2, 1)

    def forward(self, batch, conformers_index, node_index):
        x: torch.Tensor = self.node_embeddings_model(batch.z, batch.pos, node_index)
        # x = self.molecular_regression_lin(x)
        # x = self.conformers_mean_aggr(x, conformers_index)
        x = self.conformers_mean_aggr(x, conformers_index)
        x = self.molecular_regression_lin(x)
        return x


class EmbeddingsWithGATAggregationBaryCenter(EquivAggregation):
    def __init__(
        self,
        num_conformers: int,
        batch_size: int,
        learning_rate: float,
        model_name="schnet",
        is_distributed: bool = False,
        **kwargs,
    ):
        super().__init__(num_conformers, batch_size, learning_rate, model_name, is_distributed)
        out_channels = self.node_embeddings_model.hidden_channels // 2
        self.gat_embeddings_model = EquivModelsHolder.get_model("gat", self.device, feat_dim=128)
        self.transformation_matrix_3d = Linear(out_channels, out_channels)
        self.transformation_matrix_bary = Linear(out_channels, out_channels)
        self.transformation_matrix_cov = Linear(out_channels, out_channels)
        self.molecular_regression_lin = build_mlp(out_channels)

        ## hyper-param Bary Center
        if "max_iter" in kwargs:
            self.numItermax = kwargs.get("max_iter")
        if "epsilon" in kwargs:
            self.epsilon = kwargs.get("epsilon")

    def forward_dummy(self, batch, conformers_index, node_index):
        """
        batch.z: torch.Size([946],
        batch.pos: torch.Size([946, 3]
        Emb layer: SchNetNoSum(hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50, cutoff=10.0)

        z (LongTensor): Atomic number of each atom with shape
            :obj:`[num_atoms]`.
        pos (Tensor): Coordinates of each atom with shape
            :obj:`[num_atoms, 3]`.
        batch (LongTensor, optional): Batch indices assigning each atom to
            a separate molecule with shape :obj:`[num_atoms]`.
            (default: :obj:`None`)
        data_batch (Batch, optional): Batch object with additional data, such as covalent bonds attributes --> node_index

        """

        x_3d: torch.Tensor = self.node_embeddings_model(batch.z, batch.pos, node_index)
        x_3d = self.transformation_matrix_3d(x_3d)
        x_covalent = self.gat_embeddings_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

    def forward(self, batch, conformers_index, node_index):
        """
        batch.z: torch.Size([946],
        batch.pos: torch.Size([946, 3]
        Emb layer: SchNetNoSum(hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50, cutoff=10.0)

        z (LongTensor): Atomic number of each atom with shape
            :obj:`[num_atoms]`.
        pos (Tensor): Coordinates of each atom with shape
            :obj:`[num_atoms, 3]`.
        batch (LongTensor, optional): Batch indices assigning each atom to
            a separate molecule with shape :obj:`[num_atoms]`.
            (default: :obj:`None`)
        data_batch (Batch, optional): Batch object with additional data, such as covalent bonds attributes --> node_index

        """

        # x_3d: torch.Tensor = self.node_embeddings_model(batch.z, batch.pos, node_index)

        x_3d, x_bary = self.node_embeddings_model.forward_w_barycenter(
            z=batch.z,
            pos=batch.pos,
            num_conformers=self.num_conformers,
            batch=node_index,
            max_iter=self.numItermax,
            epsilon=self.epsilon,
        )
        if x_3d.get_device() >= 0:
            x_bary = x_bary.to(x_3d.get_device())
        x_bary = self.transformation_matrix_bary(x_bary)
        x_3d = self.transformation_matrix_3d(x_3d)
        x_covalent = self.gat_embeddings_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        x_covalent = self.transformation_matrix_cov(x_covalent)
        w1 = 0.2
        x = x_3d + x_covalent + w1 * x_bary
        x = self.conformers_mean_aggr(x, conformers_index)
        x = self.molecular_regression_lin(x)
        return x


class EmbeddingsWithGATAggregation(EquivAggregation):
    def __init__(
        self,
        num_conformers: int,
        batch_size: int,
        learning_rate: float,
        model_name="schnet",
        is_distributed: bool = False,
    ):
        super().__init__(num_conformers, batch_size, learning_rate, model_name, is_distributed)
        out_channels = self.node_embeddings_model.hidden_channels // 2
        self.gat_embeddings_model = EquivModelsHolder.get_model("gat", self.device, feat_dim=128)
        self.transformation_matrix_3d = Linear(out_channels, out_channels)
        self.transformation_matrix_cov = Linear(out_channels, out_channels)
        self.transformation_matrix_bary = Linear(out_channels, out_channels)
        # self.molecular_regression_lin = Linear(out_channels, 1)
        self.molecular_regression_lin = build_mlp(out_channels)
        self.readout = aggr_resolver("sum")

    def forward_dummy(self, batch, conformers_index, node_index):
        """
        batch.z: torch.Size([946],
        batch.pos: torch.Size([946, 3]
        Emb layer: SchNetNoSum(hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50, cutoff=10.0)

        z (LongTensor): Atomic number of each atom with shape
            :obj:`[num_atoms]`.
        pos (Tensor): Coordinates of each atom with shape
            :obj:`[num_atoms, 3]`.
        batch (LongTensor, optional): Batch indices assigning each atom to
            a separate molecule with shape :obj:`[num_atoms]`.
            (default: :obj:`None`)
        data_batch (Batch, optional): Batch object with additional data, such as covalent bonds attributes --> node_index

        """

        x_3d: torch.Tensor = self.node_embeddings_model(batch.z, batch.pos, node_index)
        x_3d = self.transformation_matrix_3d(x_3d)
        x_covalent = self.gat_embeddings_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

    def forward(self, batch, conformers_index, node_index):
        """
        batch.z: torch.Size([946],
        batch.pos: torch.Size([946, 3]
        Emb layer: SchNetNoSum(hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50, cutoff=10.0)

        z (LongTensor): Atomic number of each atom with shape
            :obj:`[num_atoms]`.
        pos (Tensor): Coordinates of each atom with shape
            :obj:`[num_atoms, 3]`.
        batch (LongTensor, optional): Batch indices assigning each atom to
            a separate molecule with shape :obj:`[num_atoms]`.
            (default: :obj:`None`)
        data_batch (Batch, optional): Batch object with additional data, such as covalent bonds attributes --> node_index

        """

        x_3d: torch.Tensor = self.node_embeddings_model(batch.z, batch.pos, node_index)
        x_3d = self.transformation_matrix_3d(x_3d)
        x_covalent = self.gat_embeddings_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        x_covalent = self.transformation_matrix_cov(x_covalent)
        x = x_3d + x_covalent
        x = self.conformers_mean_aggr(x, conformers_index)
        x = self.molecular_regression_lin(x)
        return x


class EmbeddingsWithGATAggregationClassification(EquivAggregationClassification):
    def __init__(
        self,
        num_conformers: int,
        batch_size: int,
        learning_rate: float,
        model_name="schnet",
        class_weights=None,
        is_distributed: bool = False,
        trade_off: bool = False,
    ):
        super().__init__(
            num_conformers,
            batch_size,
            learning_rate,
            model_name,
            class_weights,
            is_distributed,
            trade_off,
        )
        out_channels = self.node_embeddings_model.hidden_channels // 2
        self.gat_embeddings_model = EquivModelsHolder.get_model("gat", self.device, feat_dim=512)
        self.transformation_matrix_3d = Linear(out_channels, out_channels)
        self.transformation_matrix_cov = Linear(out_channels, out_channels)
        self.transformation_matrix_bary = Linear(out_channels, out_channels)
        self.molecular_regression_lin = build_mlp_class(out_channels, is_complex=True)
        ## using self-attention
        self.self_attention = SelfAttention(out_channels)

    def forward_dummy(self, batch, conformers_index, node_index):
        x_3d: torch.Tensor = self.node_embeddings_model(batch.z, batch.pos, node_index)
        x_3d = self.transformation_matrix_3d(x_3d)

        x_covalent = self.gat_embeddings_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        x_covalent = self.transformation_matrix_cov(x_covalent)

    def forward(self, batch, conformers_index, node_index):
        x_3d: torch.Tensor = self.node_embeddings_model(batch.z, batch.pos, node_index)
        x_3d = self.transformation_matrix_3d(x_3d)

        x_covalent = self.gat_embeddings_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        x_covalent = self.transformation_matrix_cov(x_covalent)

        x = x_3d + x_covalent  ## << aggregation is here
        # x = torch.cat([x_3d, x_covalent], dim=1)

        x = torch.unsqueeze(x, dim=1)
        x = self.self_attention(x)
        x = torch.squeeze(x, dim=1)

        x = self.conformers_mean_aggr(x, conformers_index)
        x = self.molecular_regression_lin(x)
        x = torch.sigmoid(x)

        return x


class EmbeddingsWithGATAggregationClassificationBaryCenter(EquivAggregationClassification):
    def __init__(
        self,
        num_conformers: int,
        batch_size: int,
        learning_rate: float,
        model_name="schnet",
        class_weights=None,
        is_distributed: bool = False,
        trade_off: bool = False,
        **kwargs,
    ):
        super().__init__(
            num_conformers,
            batch_size,
            learning_rate,
            model_name,
            class_weights,
            is_distributed,
            trade_off,
        )
        out_channels = self.node_embeddings_model.hidden_channels // 2
        self.gat_embeddings_model = EquivModelsHolder.get_model("gat", self.device, feat_dim=512)
        self.transformation_matrix_3d = Linear(out_channels, out_channels)
        self.transformation_matrix_cov = Linear(out_channels, out_channels)
        self.transformation_matrix_bary = Linear(out_channels, out_channels)
        # self.molecular_regression_lin = Linear(out_channels, 1)     ## << out_channels=64
        ## build MLP to enhance the classification problem
        # self.molecular_regression_lin = build_mlp(out_channels, is_complex=True)
        self.molecular_regression_lin = build_mlp_class(out_channels, is_complex=True)
        self.self_attention = SelfAttention(out_channels)

    def forward_dummy(self, batch, conformers_index, node_index):
        x_3d: torch.Tensor = self.node_embeddings_model(batch.z, batch.pos, node_index)
        x_3d = self.transformation_matrix_3d(x_3d)

        x_covalent = self.gat_embeddings_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        x_covalent = self.transformation_matrix_cov(x_covalent)

    def forward(self, batch, conformers_index, node_index):
        x_3d, x_bary = self.node_embeddings_model.forward_w_barycenter(
            z=batch.z, pos=batch.pos, num_conformers=self.num_conformers, batch=node_index
        )
        if x_3d.get_device() >= 0:
            x_bary = x_bary.to(x_3d.get_device())
        x_3d = self.transformation_matrix_3d(x_3d)
        x_bary = self.transformation_matrix_bary(x_bary)

        x_covalent = self.gat_embeddings_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        x_covalent = self.transformation_matrix_cov(x_covalent)
        w = 0.2
        x = x_3d + x_covalent + w * x_bary  ## << aggregation is here

        # x = torch.unsqueeze(x, dim=1)
        # x = self.self_attention(x)
        # x = torch.squeeze(x, dim=1)

        x = self.conformers_mean_aggr(x, conformers_index)

        x = self.molecular_regression_lin(x)
        x = torch.sigmoid(x)

        return x


class EmbeddingsVisualizationBaryCenter(EquivAggregationClassification):
    def __init__(
        self,
        num_conformers: int,
        batch_size: int,
        learning_rate: float,
        model_name="schnet",
        class_weights=None,
        is_distributed: bool = False,
        trade_off: bool = False,
    ):
        super().__init__(
            num_conformers,
            batch_size,
            learning_rate,
            model_name,
            class_weights,
            is_distributed,
            trade_off,
        )
        out_channels = self.node_embeddings_model.hidden_channels // 2
        self.gat_embeddings_model = EquivModelsHolder.get_model("gat", self.device, feat_dim=512)
        self.transformation_matrix_3d = Linear(out_channels, out_channels)
        self.transformation_matrix_cov = Linear(out_channels, out_channels)
        # self.molecular_regression_lin = Linear(out_channels, 1)     ## << out_channels=64
        ## build MLP to enhance the classification problem
        self.molecular_regression_lin = build_mlp(out_channels, is_complex=True)
        self.self_attention = SelfAttention(out_channels)

    def forward_dummy(self, batch, conformers_index, node_index):
        x_3d: torch.Tensor = self.node_embeddings_model(batch.z, batch.pos, node_index)
        x_3d = self.transformation_matrix_3d(x_3d)

        x_covalent = self.gat_embeddings_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        x_covalent = self.transformation_matrix_cov(x_covalent)

    def forward(self, batch, conformers_index, node_index, conformers):
        x_3d, x_bary = self.node_embeddings_model.forward_w_barycenter_visualization(
            z=batch.z,
            pos=batch.pos,
            num_conformers=self.num_conformers,
            batch=node_index,
            conformers=conformers,
        )


class CovalentEmbeddingsAggregation(EquivAggregation):
    def __init__(
        self,
        num_conformers: int,
        batch_size: int,
        learning_rate: float,
        model_name="schnet_covalent",
        is_distributed: bool = False,
    ):
        super().__init__(num_conformers, batch_size, learning_rate, model_name, is_distributed)
        self.molecular_regression_lin = Linear(self.node_embeddings_model.hidden_channels // 2, 1)

    def forward(self, batch, conformers_index, node_index):
        x: torch.Tensor = self.node_embeddings_model(batch.z, batch.pos, node_index, batch)
        x = self.molecular_regression_lin(x)
        x = self.conformers_mean_aggr(x, conformers_index)
        return x


class AttentionEmbeddingsAggregation(EquivAggregation):
    def __init__(
        self,
        num_conformers: int,
        batch_size: int,
        learning_rate: float,
        model_name="schnet",
        is_distributed: bool = False,
    ):
        super().__init__(num_conformers, batch_size, learning_rate, model_name, is_distributed)

        self.mol_embed_dim = self.node_embeddings_model.hidden_channels // 2

        # Attention matrices
        self.queries = Linear(self.mol_embed_dim, self.mol_embed_dim)
        self.keys = Linear(self.mol_embed_dim, self.mol_embed_dim)
        self.softmax = Softmax(dim=1)
        self.values = Linear(self.mol_embed_dim, self.mol_embed_dim)

        self.conformers_sum_agreggation = aggr.SumAggregation()

        self.molecular_regression_lin = Linear(self.mol_embed_dim, 1)

    def forward(self, batch, conformers_index, node_index):
        x: torch.Tensor = self.node_embeddings_model(batch.z, batch.pos, node_index)
        x = self.dot_product_attention(x)
        x = self.conformers_mean_aggr(x, conformers_index)
        x = self.molecular_regression_lin(x)
        return x

    def dot_product_attention(self, x: torch.Tensor):
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)
        sim_scores = self.softmax(torch.matmul(queries, keys.T))
        return torch.matmul(sim_scores, values)


class CovalentAttentionEmbeddingsAggregation(AttentionEmbeddingsAggregation):
    def __init__(
        self,
        num_conformers: int,
        batch_size: int,
        learning_rate: float,
        model_name="covalent_schnet_no_sum",
    ):
        super().__init__(num_conformers, batch_size, learning_rate, model_name)

    def forward(self, batch, conformers_index, node_index):
        x: torch.Tensor = self.node_embeddings_model(batch.z, batch.pos, node_index, batch)
        x = self.dot_product_attention(x)
        x = self.conformers_mean_aggr(x, conformers_index)
        x = self.molecular_regression_lin(x)
        return x


class EmbeddingsWithGAT(EquivAggregation):
    """ """

    def __init__(
        self,
        num_conformers: int,
        batch_size: int,
        learning_rate: float,
        model_name="schnet",
        is_distributed: bool = False,
    ):
        super().__init__(num_conformers, batch_size, learning_rate, model_name, is_distributed)
        out_channels = self.node_embeddings_model.hidden_channels // 2
        self.gat_embeddings_model = EquivModelsHolder.get_model("gat", self.device, feat_dim=128)
        self.molecular_regression_lin = build_mlp(out_channels)

    def forward(self, batch, conformers_index, node_index):
        """
        batch.z: torch.Size([946],
        batch.pos: torch.Size([946, 3]
        Emb layer: SchNetNoSum(hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50, cutoff=10.0)

        z (LongTensor): Atomic number of each atom with shape
            :obj:`[num_atoms]`.
        pos (Tensor): Coordinates of each atom with shape
            :obj:`[num_atoms, 3]`.
        batch (LongTensor, optional): Batch indices assigning each atom to
            a separate molecule with shape :obj:`[num_atoms]`.
            (default: :obj:`None`)
        data_batch (Batch, optional): Batch object with additional data, such as covalent bonds attributes --> node_index

        """

        x_covalent = self.gat_embeddings_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        x = self.molecular_regression_lin(x_covalent)

        return x
