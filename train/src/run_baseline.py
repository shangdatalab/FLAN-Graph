import datetime
import numpy as np
import pickle
import re
from tqdm import tqdm
import yaml
# metrics
from sklearn import metrics  # accuracy measure
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
import dgl
from dgl.dataloading import GraphDataLoader
import torch
import torch.nn as nn

from data import NodeLevelGraph
from models import get_model

from utils import get_training_weight
from utils import load_config
from utils import send_graph_to_device
from utils import parse_args

import copy
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from execute import Execute

class Execute_baseline(Execute):
    def __init__(self, loader_train, loader_val, loader_test, config):
        super().__init__(loader_train, loader_val, loader_test, config, is_gpa = False)

    def get_scores(self, model, type):
        Y_pred = []
        Y_score = []
        self.model.eval()
        Y_true = []
        if type == 'val':
            dataset = self.loader_val
        elif type == 'test':
            dataset = self.loader_test
        else:
            raise NotImplementedError()

        with torch.no_grad():
            print('-'*40, f"evaluating on {type} set", '-'*40)
            num_loop_graph = 0
            for step, batch in tqdm(enumerate(dataset), total=len(dataset)):
                summary_mask = batch.ndata['not_summary'] == 1
                if config["from_root"]:
                    batch = batch.reverse(copy_ndata=True, copy_edata=True)
                batch = send_graph_to_device(batch, DEVICE)
                if self.config["model"] == "treeLSTM":
                    try:
                        dgl.traversal.topological_nodes_generator(batch)
                    except:
                        print(f"detected loop in the graph, {num_loop_graph}/{step}\n")
                        num_loop_graph += 1
                        continue
                logits, _ = model(batch, batch.ndata['x'].squeeze(1), batch.ndata['feature'].squeeze(1))
                logits = logits.squeeze().detach().cpu().numpy()
                logits = logits[summary_mask]
                for i in np.argmax(logits, axis=1):
                    Y_pred.append(i)
                for i in logits:
                    Y_score.append(i[1])
                Y_true.extend(batch.ndata['y'][summary_mask].detach().cpu())
        return np.array(Y_pred), np.array(Y_score), np.array(Y_true)

    def train_step(self):
        length = len(self.loader_train)
        for _, batch in tqdm(enumerate(self.loader_train), total = length):
            summary_mask = batch.ndata['not_summary'] == 1
            if not config["from_root"]:
                batch = batch.reverse(copy_ndata=True, copy_edata=True)
            batch = send_graph_to_device(batch, DEVICE)
            logits, logits_copy = self.model(
                batch,
                batch.ndata['x'].squeeze(1),
                batch.ndata['feature'].squeeze(1),
            ) #future work to remove squeeze
            
            logp = F.log_softmax(logits, 1)

            loss = F.nll_loss(logp[summary_mask], batch.ndata['y'][summary_mask], reduction='sum', weight = self.weight)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # not computing training auc
            # pred = torch.argmax(logits[summary_mask], 1)
            # y_true.extend(batch.ndata['y'][summary_mask].detach().cpu())
            # for i in logits[summary_mask].squeeze().detach().cpu().numpy():
            #     y_score.append(i[1])

if __name__ == '__main__':
    # args
    args = parse_args()
    config_file_path = args.config
    config = load_config(config_file_path)

    train_dataset = NodeLevelGraph(config["train_data_graph_path"])
    val_dataset = NodeLevelGraph(config["val_data_graph_path"])
    test_dataset = NodeLevelGraph(config["test_data_graph_path"])
    execute = Execute_baseline(train_dataset, val_dataset, test_dataset, config)
    execute.train()
