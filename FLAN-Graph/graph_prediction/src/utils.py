import argparse
import yaml
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
import numpy as np
import random
import dgl

def load_config(config_file_path : str):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_training_weight(data):
    df = pd.read_csv(data, low_memory = True, delimiter = '\t')
    y = df[df["dataset"] == "train"]["claim_label_102"].values
    class_weights = compute_class_weight(
            "balanced", classes=np.unique(y), y=y
        )
    return class_weights


def send_graph_to_device(g, device):
    # nodes
    g = g.to(device)
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True)
    
    # edges
    labels = g.edge_attr_schemes()
    for l in labels.keys():
        g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)
    return g

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dgl.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config_baseline.yaml')
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()
