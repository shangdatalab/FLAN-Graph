# Author: Zhaoyi Hou (Joey), Yifei Ning (Couson)
# Last Update: 3/18/2022

import torch
import torch.nn as nn
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

import pickle
import os
from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    random_split,
    RandomSampler,
    SequentialSampler,
)
def data_prep(data, patent_class=None, noForeign=False):
        """
        Prepare dataloaders for training specified model
            - raw data - after exploding
                - File path: “/data4/xuanyu/full_data_2021/raw/”
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
        num_feat = ["similarity_product", "max_score_y", "max_score_x", "mean_score", "max_citations", "max_other_citations", "max_article_citations", "lexical_diversity"]
        cat_feat = ["patent_class", "applicantCitedExaminerReferenceIndicatorCount", "component", "transitional_phrase"]
        curr_label = "claim_label_102"
        # dataloader dir

        # [REMOVED] Randomly sample 30000 applications for training - TODO: save app num
        # seleced_app_num = set(list(train_df.application_number.unique()))
        # if train_df.application_number.nunique() > self.config['max_app_num']:
        #     seleced_app_num = set(random.sample(list(train_df.application_number.unique()), self.config['max_app_num']))
        # print(len(seleced_app_num))
        # train_df = train_df.loc[train_df.application_number.apply(lambda n: n in seleced_app_num)]

        # features to use
        num_feat = (
            num_feat
        )  # ['bp_4_skewness', 'bp_3_skewness', 'lexical_diversity'] + citation_feat
        cat_feat = cat_feat  # ['patent_class']
        df = pd.read_csv(data, low_memory = True, sep = '\t')
        print("finished reading csv")
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
        cat_trans = Pipeline(
            steps=[("input", SimpleImputer(strategy='most_frequent')),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        col_trans = ColumnTransformer(
            transformers=[
                ("num", num_trans, num_feat),
                ("cat", cat_trans, cat_feat),
            ]
        )

        # if self.use_hinge_loss:
        #     train_df = calculate_hinge_constraint(train_df, 'similarity', self.hinge_constraint_factor)
        
        
        train_df = df[df['dataset'] == 'train']

        col_trans.fit(train_df[num_feat + cat_feat])
        print("finished fitting transform")
        # preparing train dataset pickle
        test_df = df[df['dataset'] == 'test']

        app_feats_train = col_trans.transform(train_df[num_feat + cat_feat])
        app_feats_test = col_trans.transform(test_df[num_feat + cat_feat])
        print("finished applying transform")
        try:
            app_feats_train = app_feats_train.todense()
            app_feats_test = app_feats_test.todense()
        except:
            pass

        app_feats_train = torch.tensor([np.array(i).flatten() for i in app_feats_train], dtype=torch.float32)
        app_feats_test = torch.tensor([np.array(i).flatten() for i in app_feats_test], dtype=torch.float32)
        # app_feats_sample = torch.tensor([np.array(app_feats[0]).flatten()])
        train_app_nums = torch.tensor([int(i) for i in train_df.applicationNumber.values])
        test_app_nums = torch.tensor([int(i) for i in test_df.applicationNumber.values])

        train_claim_idx = torch.tensor([int(i) for i in train_df.claim_idx.values])
        test_claim_idx = torch.tensor([int(i) for i in test_df.claim_idx.values])

        train_y = torch.tensor(train_df[curr_label].values)
        test_y = torch.tensor(test_df[curr_label].values)

        

        return group_by_app_num(zip(train_df["claim_input"], train_app_nums, app_feats_train, train_claim_idx, train_y)),\
             group_by_app_num(zip(test_df["claim_input"], test_app_nums, app_feats_test, test_claim_idx, test_y)), 

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
