import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import re
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    random_split,
    RandomSampler,
    SequentialSampler,
)
import argparse
import yaml

def data_prep(data, patent_class=None, noForeign=False):
    """
    Prepare dataloaders for training specified model
        - raw data - after exploding
            - File path: “”
                - Features:
                    "dataset": "train" or "val" or "test" or “train_expanded”
                    "name": claim texts
                    curr_label: trainer_config.curr_label (very likely is "claim_label_102")
        - Config
            - Model_name
    """
    print()
    print()
    print("================== Running Data Preparation ==================")
    print()
    num_feat = [
        "similarity_product",
        "max_score_y",
        "max_score_x",
        "mean_score",
        "max_citations",
        "max_other_citations",
        "max_article_citations",
        "lexical_diversity",
    ]
    cat_feat = [
        "patent_class",
        "applicantCitedExaminerReferenceIndicatorCount",
        "component",
        "transitional_phrase",
    ]
    curr_label = "claim_label_102"

    # features to use
    num_feat = num_feat  # ['bp_4_skewness', 'bp_3_skewness', 'lexical_diversity'] + citation_feat
    cat_feat = cat_feat  # ['patent_class']
    # use chunksize to speed up
    df = pd.read_csv(data, low_memory=True, sep="\t")

    print("==>finished reading csv")

    tmp_use_app_feats = False

    # Check if all the required features are presented in the train_df
    for feat in num_feat + cat_feat:
        if feat not in df.columns:
            print("==> [Error!!] Some required features are missing!")
            print("     EXPECTED FEATURES:", (num_feat + cat_feat))
            print("     EXISTING FEATURES:", list(df.columns))
            print("     STOP!!")
            return 0

    # Pre-processing pipeline
    num_trans = Pipeline(steps=[("scaler", StandardScaler())])
    print("==>start to transform columns")

    ### ###
    cat_trans = Pipeline(
        steps=[
            ("input", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    ### ###
    col_trans = ColumnTransformer(
        transformers=[
            ("num", num_trans, num_feat),
            ("cat", cat_trans, cat_feat),
        ]
    )

    print("=====>start fitting transform")
    print("=====> size (#claims) : {}".format(len(df.index)))


    col_trans.fit(df[num_feat + cat_feat])
    print("=====>finished fitting transform")
    # preparing train dataset pickle

    print("=====>start to apply transform")
    app_feats = col_trans.transform(df[num_feat + cat_feat])
    print("==>finished transform columns")
    try:
        app_feats = app_feats.todense()
    except:
        pass

    app_feats = torch.tensor(
        [np.array(i).flatten() for i in app_feats], dtype=torch.float32
    )
    app_nums = torch.tensor([int(i) for i in df.applicationNumber.values])
    claim_idx = torch.tensor([int(i) for i in df.claim_idx.values])

    y = torch.tensor(df[curr_label].values)
    return group_by_app_num(
            zip(
                df["claim_input"],
                app_nums,
                app_feats,
                claim_idx,
                y,
            )
        )

def group_by_app_num(list_of_tuple):
    lastAppNum = 0
    ans = []
    cur = []
    for claim_text, app_num, app_feat, original_claim_idx, y in list_of_tuple:
        if lastAppNum != 0 and lastAppNum != app_num:
            ans.append(cur)
            cur = []
        lastAppNum = app_num
        cur.append([claim_text, app_num, app_feat, original_claim_idx, y])
    ans.append(cur)
    return ans


def get_training_weight(data):
    class_weights = compute_class_weight("balanced", classes=np.unique(data), y=data.numpy())
    print("weight is " + str(class_weights))
    return class_weights

def group_edge_type(edge_type):
    d = {
        1:0,
        2:1,
        3:2,

        4:8,
        5:9,

        10:3,
        11:3,
        12:3,
        13:3,
        14:3,
        15:3,
        16:3,
        17:3,
        18:3,

        20:4,
        21:4,
        22:4,

        30:5,

        40:6,
        41:6,

        50:7
    }
    return d[edge_type]
    # return 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--graph_path', type=str)
    parser.add_argument('--root2child', dest='root2child', action='store_true')
    parser.add_argument('--no-root2child', dest='root2child', action='store_false')
    parser.set_defaults(root2child=True)
    return parser.parse_args()