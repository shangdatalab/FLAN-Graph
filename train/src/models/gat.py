"""GAT using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/GAT
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv
from .gnn_base import GNNBase
import dgl

class GAT(GNNBase):
    def __init__(self,
                 x_size,
                 feature_size,
                 activation,
                 dropout,
                 is_gpa,
                 config,
                 **kwargs):
        super(GAT, self).__init__(is_gpa, config)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATConv(x_size, self.hidden_size, self.num_heads, activation=activation, allow_zero_in_degree = True))
        # hidden layers
        for _ in range(self.gnn_layers):
            self.layers.append(GATConv(self.hidden_size * self.num_heads , self.hidden_size, self.num_heads, activation=activation, allow_zero_in_degree = True))
        # output layer
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.hidden_size * self.num_heads + feature_size, 2) # 2 classes, use weight

    def get_hidden_for_classifier(self, g, x):
        h = x #right now shape (nodes, 1, feat)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            h = h.view(h.size(0), -1) #concatenate multihead feature
        # breakpoint()
        return self.dropout(h)
