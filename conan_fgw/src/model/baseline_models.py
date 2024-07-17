from torch.nn import Linear

from conan_fgw.src.model.common import EquivAggregation


class GATEmbeddingAggregation(EquivAggregation):
    def __init__(
        self, num_conformers: int, batch_size: int, learning_rate: float, model_name="gat"
    ):
        super().__init__(num_conformers, batch_size, learning_rate, model_name)
        self.molecular_regression_lin = Linear(64, 1)

    def forward(self, batch, conformers_index, node_index):
        x = self.node_embeddings_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        x = self.conformers_mean_aggr(x, conformers_index)
        x = self.molecular_regression_lin(x)
        return x
