import pickle
import dgl
import pandas as pd
import torch
from dgl.data import DGLDataset
from tqdm import tqdm
from dgl.data.utils import save_graphs, load_graphs
from .utils import group_edge_type
import os

class GraphLevelGraph(DGLDataset):
    def __init__(self,
                path,
                claims_list_transformed = None,
                url=None,
                raw_dir=None,
                force_reload=False,
                verbose=False,
                root2child=True
                ):
        self.claims_list_transformed = claims_list_transformed
        self.path = path
        self.root2child = root2child
        super(GraphLevelGraph, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        # download raw data to local disk
        pass

    def process(self):
        # process raw data to graphs, labels, splitting masks
        pass

    def __getitem__(self, idx):
        # get one example by index
        if self.root2child == True:
            self.trees[idx] = dgl.reverse(self.trees[idx], copy_edata=True)
        return self.trees[idx], self.claim_info_dicts["app_num"][idx], self.claim_info_dicts["feat_encoding"][idx], self.claim_info_dicts["original_claim_idx"][idx], self.claim_info_dicts["label"][idx], 

    def __len__(self):
        # number of data examples
        return len(self.trees)

    def load(self):
        # load processed data from directory `self.save_path`
        print("loading %s" %(self.path))
        self.trees, self.claim_info_dicts = load_graphs(self.path)
        print("finish loading %s" %(self.path))
        # add edge type infomration from dict
        try:
            for i in range(self.__len__()):
                self.trees[i].edata["edge_type"] = torch.tensor([group_edge_type(edge_type.item()) for edge_type in self.trees[i].edata["edge_type"]])
            print("finish loading edge_type")
        except:
            print("ERROR while loading edge_type!")
 

    def has_cache(self):
        print(f"path {self.path} exists {os.path.exists(self.path)}")
        return os.path.exists(self.path)

    def x_size(self):
        return self.trees[0].ndata['encoding'].shape[1]

    def feat_size(self):
        return self.claim_info_dicts['feat_encoding'].shape[1]
    
    def get_labels(self):
        return self.claim_info_dicts['label']
    
