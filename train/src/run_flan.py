import datetime
import numpy as np
import pickle
import re
from tqdm import tqdm
# metrics
from sklearn import metrics  # accuracy measure
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
import dgl
from dgl.dataloading import GraphDataLoader
import torch
import torch.nn as nn

from data import GraphLevelGraph
from models import get_model

from utils import get_training_weight
from utils import load_config
from utils import send_graph_to_device
from utils import parse_args

import torch.nn.functional as F
import copy
from execute import Execute

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Execute_GPA(Execute):
    def __init__(self, loader_train, loader_val, loader_test, config):
        super().__init__(loader_train, loader_val, loader_test, config,  is_gpa = True)

    def get_scores(self, model, dataset_type):
        model.eval()
        Y_pred, Y_score, Y_true, app_nums = ([] for i in range(4))
        if dataset_type == 'val':
            dataset = self.loader_val
        elif dataset_type == 'test':
            dataset = self.loader_test
        else:
            raise NotImplementedError()

        with torch.no_grad():
            print("-" * 40, f"evaluating on {dataset_type} set", "-" * 40)
            for step, batch in tqdm(
                enumerate(dataset), total=len(dataset)
            ):
                batch = self.send_batch_to_device(batch, DEVICE)
                graph, app_num, feat_encoding, original_claim_idx, label = batch
                logits, logits_hinge = model(graph,  graph.ndata['encoding'], feat_encoding)
                logits = logits.squeeze().detach().cpu().numpy()
                app_nums.extend(list(zip(app_num.cpu().numpy(),original_claim_idx.cpu().numpy())))

                for i in np.argmax(logits, axis=1):
                    Y_pred.append(i)
                for i in logits:
                    Y_score.append(i[1])
                Y_true.extend(label.cpu().numpy())
        return np.array(Y_pred), np.array(Y_score), np.array(Y_true)

    def train_step(self):
        length = len(self.loader_train)
        for step, batch in tqdm(enumerate(self.loader_train), total=length):
            batch = self.send_batch_to_device(batch, DEVICE)
            graph, app_num, feat_encoding, original_claim_idx, label = batch
            logits, _ = self.model(
                graph, 
                graph.ndata['encoding'], 
                feat_encoding,
                )
            logits = logits.squeeze(1)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(
                logp,
                label,
                reduction="sum",
                weight=self.weight,
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        

if __name__ == "__main__":
    # args
    args = parse_args()
    config_file_path = args.config
    config = load_config(config_file_path)
    # train
    train_dataset = GraphLevelGraph(config["train_data_graph_path"])
    val_dataset = GraphLevelGraph(config["val_data_graph_path"])
    test_dataset = GraphLevelGraph(config["test_data_graph_path"])
    execute = Execute_GPA(train_dataset, val_dataset, test_dataset, config)
    execute.train()
