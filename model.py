import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data_global):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        self.gru = nn.GRU(1, hidden_channels, batch_first=True)
        for node_type in data_global.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data_global.metadata(), num_heads)
            self.convs.append(conv)

        self.aavgpool = nn.AdaptiveAvgPool1d(output_size=5)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['flow'])