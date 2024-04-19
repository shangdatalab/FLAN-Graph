from .gat import GAT
from .gcn import GCN
from .sage import SAGE
from .gcn2conv import Graph2Conv
from .treeLSTM import TreeLSTM
import torch.nn.functional as F

def get_model(config: dict, x_size: int, feature_size: int, is_gpa: bool):
    model = None
    if config["model"] == "gat":
        model = GAT
    elif config["model"] == "gcn":
        model = GCN
    elif config["model"] == "sage":
        model = SAGE
    elif config["model"]== "gcn2":
        model = Graph2Conv
    elif config["model"] == "treeLSTM":
        model = TreeLSTM
    else:
        raise Exception("model not found")
    return model(
        x_size = x_size,
        feature_size = feature_size,
        activation = F.relu,
        dropout = 0.2,
        is_gpa = is_gpa,
        config = config
    )