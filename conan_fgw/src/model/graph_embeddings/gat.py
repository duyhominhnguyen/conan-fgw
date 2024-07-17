import torch
from torch_geometric.nn import GATConv, aggr


class GATBased(torch.nn.Module):
    def __init__(self, out_channels: int = 64, edge_dim=3):
        super().__init__()

        self.gat_conv1 = GATConv(in_channels=-1, out_channels=out_channels, edge_dim=edge_dim)
        self.gat_conv2 = GATConv(
            in_channels=out_channels, out_channels=out_channels, edge_dim=edge_dim
        )
        self.node_aggregation = aggr.SumAggregation()
        # self.dense_adj = ToDense()

    def forward(self, x, edge_index, edge_attr, batch):
        """
        x: node features [num_nodes, feat_size]
        """
        x = x.float()
        edge_attr = edge_attr.float()
        h = self.gat_conv1(x, edge_index, edge_attr)
        h = self.gat_conv2(h, edge_index, edge_attr)
        h = self.node_aggregation(h, batch)
        return h
