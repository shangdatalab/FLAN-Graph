"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from .gnn_base import GNNBase

class GCN(GNNBase):
    def __init__(self,
                 x_size,
                 feature_size,
                 activation,
                 dropout,
                 is_gpa,
                 config,
                 **kwargs):
        super(GCN, self).__init__(is_gpa, config)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(x_size, self.hidden_size , activation=activation, allow_zero_in_degree = True))
        # hidden layers
        for i in range(self.gnn_layers):
            self.layers.append(GraphConv(self.hidden_size , self.hidden_size , activation=activation, allow_zero_in_degree = True))
        # output layer
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.hidden_size + feature_size, 2) # 2 classes, use weight


    def get_hidden_for_classifier(self, g, x):
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        
        return self.dropout(h)
