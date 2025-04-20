import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm

class GraphSAGE(nn.Module):
    """
    GraphSAGE model with configurable layers, dropout, and optional batch normalization.
    
    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden units in intermediate layers.
        out_channels (int): Number of output features per node (e.g., number of classes).
        num_layers (int): Total number of GraphSAGE layers (minimum: 2).
        dropout (float): Dropout rate applied after ReLU activation.
        use_batchnorm (bool): Whether to apply BatchNorm between layers.
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, use_batchnorm=True):
        super(GraphSAGE, self).__init__()

        assert num_layers >= 2, "num_layers must be at least 2"
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        if use_batchnorm:
            self.norms.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if use_batchnorm:
                self.norms.append(BatchNorm(hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                if self.use_batchnorm:
                    x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x