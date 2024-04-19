"""GAT using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/GAT
"""
import torch
import torch.nn as nn
import dgl

class GNNBase(nn.Module):

    def __init__(self, is_gpa, config):
        super(GNNBase, self).__init__()
        self.is_gpa = is_gpa
        self.config = config
        self.use_graph_feature = config["use_graph_feature"]
        self.use_app_feats = config["use_app_feats"]
        self.gnn_aggregate_type = config.get("gnn_aggregate_type", None) #None for coarse, not used
        self.aggregator_type = config["aggregator_type"]
        self.num_heads = config["num_heads"]
        self.gnn_layers = config["gnn_layers"]
        self.hidden_size = config["hidden_size"]

    def get_hidden_for_classifier(self):
        pass

    def unbatch_hidden(self, g, h, use_graph_feature = True):
        device = h.device
        graph_level_h = []
        graph_level_feat = []
        g.ndata['h'] = h
        unbatched_graphs = dgl.unbatch(g)
        for idx, claim_graph in enumerate(unbatched_graphs):
            if use_graph_feature:
                num_target_nodes = torch.count_nonzero(claim_graph.ndata["is_target"] == 1)
                num_total_nodes = torch.tensor(claim_graph.nodes().shape[0],device = device)
                graph_features = torch.stack((
                    torch.max(claim_graph.ndata["depth"]), # maximum depth
                    num_target_nodes, # num target nodes
                    num_total_nodes, # num total nodes
                ))
                graph_level_feat.append(graph_features)
            if self.gnn_aggregate_type == "sum":
                target_hidden_all = claim_graph.ndata["h"][claim_graph.ndata["is_target"].flatten()]
                target_hidden = torch.sum(target_hidden_all, 0).flatten()
            elif self.gnn_aggregate_type == "mean":
                target_hidden_all = claim_graph.ndata["h"][claim_graph.ndata["is_target"].flatten()]
                target_hidden = torch.mean(target_hidden_all, 0).flatten()
            elif self.gnn_aggregate_type == "root":
                target_hidden_all = claim_graph.ndata["h"][claim_graph.ndata["is_root"].flatten()]
                target_hidden = torch.sum(target_hidden_all, 0).flatten()
            else:
                raise NotImplementedError()

            graph_level_h.append(target_hidden)
        if use_graph_feature:
            return torch.stack(graph_level_h), torch.stack(graph_level_feat)
        else:
            return torch.stack(graph_level_h), None

    def forward(
        self,
        g,
        x,
        feature):
        h = self.get_hidden_for_classifier(g, x)
        feature_final = None
        # breakpoint()
        if self.use_app_feats:
            feature_final = feature
        if self.is_gpa:
            h, graph_feat = self.unbatch_hidden(g, h)
            if self.use_graph_feature:
                if feature_final is not None:
                    feature_final = torch.cat((feature_final, graph_feat), dim = 1)
                else:
                    feature_final = graph_feat
        if feature_final is not None:
            h = torch.cat((h, feature_final), dim=1)
        logits = self.classifier(h)

        return logits, None #forget about hinge