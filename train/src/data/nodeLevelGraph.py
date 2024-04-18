import pickle
import dgl
from dgl.data import DGLDataset
from tqdm import tqdm
import pandas as pd
from dgl.data.utils import save_graphs, load_graphs

import os

class NodeLevelGraph(DGLDataset):
    """ Build a bunch of trees

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self,
                path,
                url=None,
                raw_dir=None,
                save_dir=None,
                force_reload=False,
                verbose=False):
        self.path = path
        self.transformed_set = set()
        self.trees = None

        super(NodeLevelGraph, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        # download raw data to local disk
        pass

    def process(self):
        # process raw data to graphs, labels, splitting masks
        pass

    def __getitem__(self, idx):
        return self.trees[idx]

    def __len__(self):
        return len(self.trees)

    def save(self):
        pass

    def load(self):
        print(f"loading {self.path}")
        self.trees, _ = load_graphs(self.path)

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        print(f"path {self.path} exists {os.path.exists(self.path)}")
        return os.path.exists(self.path)

    def x_size(self):
        return self.trees[0].ndata['x'].shape[2]

    def feat_size(self):
        return self.trees[0].ndata['feature'].shape[2]