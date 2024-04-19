"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import dgl
from dgl.nn import GCN2Conv
from .gnn_base import GNNBase


class Graph2Conv(GNNBase):
    def __init__(
        self,
        x_size,
        feature_size,
        activation,
        is_gpa,
        config,
        input_alpha=0.1,
        input_lambda=1,
        dropout=0,
    ):
        super(Graph2Conv, self).__init__(is_gpa, config)
        # super(nn.Module, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(x_size, self.hidden_size, activation=activation, allow_zero_in_degree = True))

        # GCN2Conv layers
        for i in range(self.gnn_layers):
            self.layers.append(
                GCN2Conv(
                    self.hidden_size,
                    i+1,
                    alpha=input_alpha,
                    lambda_=input_lambda,
                    activation=activation,
                    allow_zero_in_degree = True
                )
            )

        # output layer
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.hidden_size + feature_size, 2)  # 2 classes, use weight


    def get_hidden_for_classifier(self, g, x):
        h = x
        h_0 = h
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = layer(g, h, h_0)
                h = self.dropout(h)
            else:
                h = layer(g, h)
                h_0 = h
        return self.dropout(h)