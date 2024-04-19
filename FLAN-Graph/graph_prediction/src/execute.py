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
from utils import set_seed
import copy
import torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Execute:
    def __init__(self, dataset_train, dataset_val, dataset_test, config, is_gpa):

        set_seed(config["seed"])

        if config["use_precomputed_class_weight"]:
            self.weight = torch.tensor([2.5844, 0.6199], dtype=torch.float).to(DEVICE)
        else:
            self.weight = torch.tensor(get_training_weight(config["data"]), dtype=torch.float).to(DEVICE)

        feature_size = 0
        if not is_gpa and config["use_graph_feature"]:
            raise Exception()

        if config["use_app_feats"]:
            feature_size += dataset_train.feat_size()
        if is_gpa and config["use_graph_feature"]:
            feature_size += 3
        self.model = get_model(config, x_size=dataset_train.x_size(), feature_size=feature_size, is_gpa = is_gpa)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        self.loader_train = GraphDataLoader(
            dataset_train,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
        )

        self.loader_val = GraphDataLoader(
            dataset_val,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
        )

        self.loader_test = GraphDataLoader(
            dataset_test,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
        )

        self.config = config
        self.cur_epoch = 0
        
        self.model.to(DEVICE)

    def train(self):
        cur_best_auc = 0.0
        best_model = copy.deepcopy(self.model)
        self.model.to(DEVICE)
        best_epoch = 1
        for epoch in range(self.config["epochs"]):
            self.cur_epoch += 1
            self.model.train()
            print("==================","training epoch {}".format(epoch), "==================")
            self.train_step()

            auc = self.evaluation(self.model, self.cur_epoch, 'val')
            if cur_best_auc < auc:
                best_model = copy.deepcopy(self.model)
                cur_best_auc =auc
                best_epoch = self.cur_epoch
        self.evaluation(best_model, best_epoch, 'test')

    def output_analysis(self, pred, targets):
        pred = np.array(pred)
        targets = np.array(targets)
        acc = sum(pred == targets) / len(pred)
        tn, fp, fn, tp = confusion_matrix(targets, pred, labels = [0,1]).ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        tnr = tn / (tn + fp)
        npv = tn / (tn + fn)

        return {
            "acc": round(100 * acc, 2),
            "precision": round(100 * precision, 2),
            "recall": round(100 * recall, 2),
            "tnr": round(100 * tnr, 2),
            "npv": round(100 * npv, 2),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    def evaluation(self, model, epoch, type = 'val'):
        Y_pred, Y_score, Y_test = self.get_scores(model, type)
        pts = np.linspace(0, 1, 101)
        f1s = []
        for pt in pts:
            f1s.append(f1_score(Y_test, Y_score > pt, average="macro"))
            
        now = datetime.datetime.now()
        with open(self.config["output_des"], "a+") as f:
            f.write(str(now))
            f.write("\nEpoch %d/%d, %s auc: %.5f, optim prauc: %.5f, negauc: %.5f, optimal macro F1 score: %.5f, from_root: %s, weighted auc: %.5f "
            % (
                epoch,
                self.config["epochs"],
                type,
                roc_auc_score(Y_test, Y_score),
                average_precision_score(Y_test, Y_score),
                average_precision_score(1 - Y_test, Y_score),
                np.max(f1s),
                str(self.config["from_root"]),
                roc_auc_score(Y_test, Y_score, average='weighted')
            ))
            config_str = yaml.dump(self.config, default_flow_style=False)
            f.write(config_str)
            f.write('\n' + str(self.output_analysis(Y_pred, Y_test)) + '\n\n')

        return roc_auc_score(Y_test, Y_score)

    def send_batch_to_device(self, batch, device):
        # nodes
        for i in range(0, len(batch)):
            batch[i] = batch[i].to(device)
        return batch

    def get_scores(self):
        pass

    def train_step(self):
        pass