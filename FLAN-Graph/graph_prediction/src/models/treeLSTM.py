from elasticsearch_dsl import Q
import numpy as np
import pandas as pd
import pickle
import re
from tqdm import tqdm

# metrics
from sklearn import metrics  # accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import dgl
import torch as th
import networkx as nx
import matplotlib.pyplot as plt
import dgl.function as fn
import torch
import torch.nn as nn
from .treeLSTMCell import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from .gnn_base import GNNBase

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TreeLSTM(GNNBase):
    def __init__(
        self,
        x_size,
        feature_size,
        activation,
        is_gpa,
        config,
        dropout = 0.2,
        **kwargs
    ):
        super(TreeLSTM, self).__init__(is_gpa, config)
        self.x_size = x_size
        self.cell = TreeLSTMCell(x_size, self.hidden_size).to(DEVICE)
        self.classifier = nn.Linear(self.hidden_size + feature_size, 2)
        self.dropout = nn.Dropout(dropout)

    def get_hidden_for_classifier(self, g, x):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.

        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        # g = batch.graph
        # feed embedding
        g.ndata["W_iou"] = self.cell.W_iou(x)
        g.ndata["W_f"] = self.cell.W_f(x)
        h = torch.randn(g.num_nodes(), self.hidden_size).to(DEVICE)
        c = torch.randn(g.num_nodes(), self.hidden_size).to(DEVICE)
        if isinstance( self.cell.U_iou, nn.ModuleList):
            g.ndata["U_iou"] = self.cell.U_iou[0](h)  # only used for the roots, children's U_iou from parent
        else:
            g.ndata["U_iou"] = self.cell.U_iou(h)
        if self.cell.b_iou.shape[0] > 1:
            g.ndata["b_iou"] = torch.vstack([self.cell.b_iou[0]]* g.num_nodes()).to(DEVICE)
        else:
            g.ndata["b_iou"] = torch.vstack([self.cell.b_iou]* g.num_nodes()).to(DEVICE)
        g.ndata[
            "c"
        ] = c  # only used for the roots, children's c will be replaced by the one from parent
        g.ndata["h"] = h
        # propagate
        try:
            generate = dgl.traversal.topological_nodes_generator(g)
        except:
            print("detected loop in the graph")

        l = [x.to(DEVICE) for x in list(generate)]
        g.prop_nodes(
            l, self.cell.message_func, self.cell.reduce_func, self.cell.apply_node_func
        )  # compute logits
        return g.ndata["h"]