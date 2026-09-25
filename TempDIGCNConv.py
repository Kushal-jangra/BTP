"""
TempDIGCNConv.py

Directed Graph Convolutional Operator with Multi-Dimensional Temporal Edge Attribute Encodings.
"""
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

class TempDIGCNConv(MessagePassing):
    """
    Temporal DiGCN Convolutional Layer.
    Aggregates baseline transition-weighted messages with a bounded temporal
    residual. The transition-count channel preserves the original OCDiGCN
    message path; standardized time features provide the novelty branch.
    """
    def __init__(self, in_channels, out_channels, edge_dim=3,
                 temporal_alpha_init=0.1, bias=False, **kwargs):
        super(TempDIGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        
        if edge_dim < 2:
            raise ValueError("TempDIGCNConv requires a count channel and temporal channels.")

        # Encode only temporal channels. The count channel remains on the
        # baseline path below, preserving transition-frequency information.
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim - 1, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels, bias=False),
        )
        # Start with a small temporal contribution so the model begins close
        # to the validated baseline and learns the temporal correction safely.
        self.temporal_alpha = Parameter(torch.tensor(float(temporal_alpha_init)))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        x = torch.matmul(x, self.weight)

        if edge_attr is None:
            raise RuntimeError("Temporal edge attributes (E_{ij}) cannot be None.")

        # Encode standardized mean/variance time features into a residual.
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        count_weight = edge_attr[:, 0]
        temporal_emb = self.edge_encoder(edge_attr[:, 1:])
        return self.propagate(
            edge_index,
            x=x,
            count_weight=count_weight,
            temporal_emb=temporal_emb,
        )

    def message(self, x_j, count_weight, temporal_emb):
        # Baseline path + bounded temporal residual. tanh prevents a single
        # extreme temporal edge from dominating the SVDD representation.
        baseline_message = count_weight.view(-1, 1) * x_j
        temporal_residual = self.temporal_alpha * torch.tanh(temporal_emb)
        return baseline_message + temporal_residual

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, edge_dim={self.edge_dim})'
