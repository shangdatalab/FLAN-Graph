import torch
import torch.nn as nn
import pickle

class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, hidden_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * hidden_size, bias=False)
        self.U_iou = nn.Linear(1 * hidden_size, 3 * hidden_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * hidden_size))

        self.U_f = nn.Linear(1 * hidden_size, 1 * hidden_size)
        self.W_f = nn.Linear(x_size, hidden_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros(1, hidden_size))

    def message_func(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        # mailbox has shape (num_nodes, num_child, feature_size
        # We perform a degree bucketing mechanism where messages for nodes with a same degree are processed together. (DGL)

        # concatenate h_jl for equation (1), (2), (3), (4)
        h_sum = torch.sum(nodes.mailbox["h"], dim=1)
        # equation (4), broadcasting b_f
        f = torch.sigmoid(
            self.U_f(nodes.mailbox["h"])
            + nodes.data["W_f"].view(nodes.data["W_f"].size(0), 1, -1)
            + self.b_f
        )
        # second term of equation (7)
        c = torch.sum(f * nodes.mailbox["c"], dim=1)
        # print('iou reduce_func: ', self.U_iou(h_cat).shape)
        return {
            "U_iou": self.U_iou(h_sum),
            "c": c,
        }  # (num_nodes, hidden_size * 3), (num_nodes, hidden_size)

    def apply_node_func(self, nodes):
        # equation (1), (3), (4)
        iou = nodes.data["W_iou"] + nodes.data["U_iou"] + self.b_iou
        iou = torch.squeeze(iou, dim=1)
        # print('iou apply_fun: ', iou.shape )
        (i, o, u) = torch.chunk(iou, 3, 1)
        # print('i: ', i.shape)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        # equation (5)
        c = i * u + nodes.data["c"]
        # equation (6)
        h = o * torch.tanh(c)

        return {"h": h, "c": c}