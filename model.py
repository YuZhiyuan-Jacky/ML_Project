import torch
from torch import nn
import torch.nn.functional as F


class BasicAdjGNN(nn.Module):
    """A simple GNN implemented with sparse adjacency matrix multiplication."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1]) for index in range(num_layers)
        )
        self.dropout = dropout

    def forward(self, data):
        x = data.x
        for index, layer in enumerate(self.layers):
            x = torch.sparse.mm(data.adj_norm, x)
            x = layer(x)
            if index != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class PyGGCN(nn.Module):
    """GCN implemented with torch_geometric.nn.GCNConv."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        super().__init__()
        from torch_geometric.nn import GCNConv

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.convs = nn.ModuleList(
            GCNConv(dimensions[index], dimensions[index + 1]) for index in range(num_layers)
        )
        self.dropout = dropout

    def forward(self, data):
        x = data.x
        for index, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            if index != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class PyGGraphTransformer(nn.Module):
    """Graph Transformer implemented with PyG TransformerConv."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        heads: int,
    ):
        super().__init__()
        from torch_geometric.nn import TransformerConv

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if heads < 1:
            raise ValueError("heads must be at least 1.")
        if num_layers > 1 and hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads for graph_transformer.")

        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(TransformerConv(input_dim, output_dim, heads=1, concat=False, dropout=dropout))
            return

        per_head_hidden = hidden_dim // heads
        self.convs.append(
            TransformerConv(input_dim, per_head_hidden, heads=heads, concat=True, dropout=dropout)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(hidden_dim, per_head_hidden, heads=heads, concat=True, dropout=dropout)
            )
        self.convs.append(TransformerConv(hidden_dim, output_dim, heads=1, concat=False, dropout=dropout))

    def forward(self, data):
        x = data.x
        for index, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            if index != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def build_model(args, input_dim: int, output_dim: int) -> nn.Module:
    if args.model == "basic_gnn":
        return BasicAdjGNN(input_dim, args.hidden_dim, output_dim, args.num_layers, args.dropout)
    if args.model == "gcn":
        return PyGGCN(input_dim, args.hidden_dim, output_dim, args.num_layers, args.dropout)
    if args.model == "graph_transformer":
        return PyGGraphTransformer(
            input_dim,
            args.hidden_dim,
            output_dim,
            args.num_layers,
            args.dropout,
            args.heads,
        )
    raise ValueError(f"Unknown model: {args.model}")
