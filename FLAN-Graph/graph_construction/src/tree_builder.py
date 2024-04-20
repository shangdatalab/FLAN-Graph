import os
import re
import sys
import dgl
import nltk
import lzma
import tqdm
import time
import json
import string
import pickle
import pymongo
import subprocess
import pandas as pd
from multiprocessing import Pool

from tqdm import tqdm

import torch as th
import torch.multiprocessing

from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy("file_system")

from nltk import Tree, CoreNLPParser
from nltk.corpus import stopwords

import networkx as nx
from networkx.readwrite import json_graph

import matplotlib.pyplot as plt

from datetime import datetime
from dateutil.relativedelta import *
from sentence_transformers import SentenceTransformer


from utils import *


# mute output from sentence transformer for
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

############################### Configuration Variables ###############################
BATCH_SIZE = 15
NUM_WORKERS = 10

# Follow this link to setup StanfordCoreNLPServer
# https://stanfordnlp.github.io/CoreNLP/download.html
PARSER_SERVER_ADD = "" # address of StanfordCoreNLPServer
PARSER_DIRECTORY = "" # path to the directory containing StanfordCoreNLPServer
PARSER_SERVER_COMMAND = 'nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -quiet true -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9196 -port 9196 -timeout 300000 > ./nohup.out'


CUDA_LIST = []


############################### Variables ###############################
DATABASE_PARAMS = ""
CSV_DATA_PATH = ""
NESTED_NUMERICAL_INDICATOR = ["(2)", "(ii)", "(b)"]  # indicator of nested component
NESTED_INDICATOR_CAMP = {"(2)": "(1)", "(ii)": "(i)", "(b)": "(a)"}
PUNCT = string.punctuation
STOPWORDS = set(stopwords.words("english"))

IDENTITY_PATTERNS = """     identiy_VBG: {<CC>*(<VBG>|<VB>)<IN>*<DT>*<VBN>*<JJ>*(((<NN>+|<NNP>+|<NNS>+)+<IN>?<DT>*<VBN>*<JJ>*(<NN>+|<NNP>+|<NNS>+)*)|(<CC>+<IN>?<DT>*<VBN>*<JJ>*(<NN>+|<NNP>+|<NNS>+)+))}
                            identity_NN: {<CC>*<DT>*<VBN>*<JJ>*(((<NN>+|<NNP>+|<NNS>+)<IN>?<DT>*<VBN>*<JJ>*(<NN>+|<NNP>+|<NNS>+)*)|(<CC>+<IN>?<DT>*<VBN>*<JJ>*(<NN>+|<NNP>+|<NNS>+)+))} """
IDENTITY_PARSER = nltk.RegexpParser(IDENTITY_PATTERNS)

# indicator of leader claims
LEADER_CONDITION_VOCAB = ["comprising", "consisting"]

# indicator to switch matching method
CLAIM_MATCHING_TOGGLE = ["wherein", "where"]
CANCEL_INDICATOR = ["(canceled)", "(previously canceled)"]

# Relations
NLL_DEFUALT_RELA = 9
LL_DEFUALT_RELA = 10
INC_DEFUALT_RELA = 50

# cross-claims edge types
CROSS_RELATIONS = {
    1: "wherein", 
    2: "whereby",
    3: "adapted for",
    4: "adapted to",
    5: "comprising",
    6: "consist",
    7: "consisting(?!\s+essentially)",
    8: "consisting essentially",
}

# within-claims edge types
INNER_RELATIONS = {
    11: "comprise",
    12: "comprising",
    13: "comprised",
    14: "include",
    15: "including",
    16: "contain",
    17: "containing",
    18: "characterized by",
    19: "consist",
    20: "consisting(?!\s+essentially)",
    21: "consisting essentially",
    22: "having",
    23: "composed of",
}

############################### Top-Level Function ###############################
def buildGraphs(claims_list_transformed, root2child=True, graph_path="./full_train_data.graph.metagraph.main.pickle"):
    global META_GRAPHS
    global ROOT2CHILD
    global GRAPH_PATH

    ROOT2CHILD = root2child
    GRAPH_PATH = graph_path

    # build metagarphs
    print("==> Start to build graphs")
    meta_graphs, skip_count = _build_meta_graphs(claims_list_transformed)
    print("!! Skiping {} applications".format(skip_count))
    
    # encode graphs
    print("==> Start to add encodings")
    encoder = SentenceTransformer("stsb-roberta-large", device="cuda")

    meta_graphs = _encode_graph_nodes(meta_graphs, encoder)

    # conert to dgl trees
    print("==> Start to converting to dgl graphs")
    META_GRAPHS = meta_graphs

    trees = []
    graph_level_infoS = []
    for meta_graph in tqdm(meta_graphs):
        dgl_to_netwx_idx = {}
        for claim_graph, graph_level_info in meta_graph:
            if claim_graph.number_of_edges() == 0:
                dgl_graph = dgl.from_networkx(
                    claim_graph, node_attrs=["encoding", "depth"]
                )
                dgl_graph.edata["edge_type"] = torch.ones(
                    dgl_graph.num_edges(), dtype=torch.int32
                )
            else:
                dgl_graph = dgl.from_networkx(
                    claim_graph,
                    node_attrs=["encoding", "depth"],
                    edge_attrs=["edge_type"],
                )

            # add note features
            for idx, tuple_idx in enumerate(claim_graph.nodes):
                dgl_to_netwx_idx[(tuple_idx)] = idx
                # add root identifier
                dgl_graph.nodes[idx].data["is_root"] = torch.tensor(
                    claim_graph.nodes[tuple_idx]["root"]
                ).reshape(1, -1)

                app_idx, claim_idx, *component_index = tuple_idx
                dgl_graph.nodes[idx].data["tuple_idx"] = torch.tensor(
                    [app_idx, claim_idx, component_index[0]]
                ).reshape(1, -1)
                dgl_graph.nodes[idx].data["is_target"] = torch.tensor(
                    claim_idx == claim_graph.true_claim_idx
                ).reshape(1, -1)

            # revert the edges
            if not root2child:
                dgl_graph = dgl.reverse(dgl_graph, copy_edata=True)

            _set_attributes( 
                dgl_graph,
                app_num=claim_graph.app_num,
                feat_encoding=claim_graph.feat_encoding,
                label=claim_graph.label,
                original_claim_idx=claim_graph.original_claim_idx,
                true_claim_idx=claim_graph.true_claim_idx,
            )
            trees.append(dgl_graph)
            graph_level_infoS.append(graph_level_info)

    print(datetime.now().time())
    return trees, graph_level_infoS

################################ Helper Functions ################################

def _build_batch_graphs(
    apps_claims_list,
    identity_map,
    leader_claims,
    pos_tagger,
    skipping_app_idx_set,
    mute_tqdm,
):
    meta_graphs = []
    skipping_count = 0

    for app_idx, app_claims in enumerate(tqdm(apps_claims_list, disable=mute_tqdm)):

        app_graph = []

        # skipping apps causing errors before
        if app_idx in skipping_app_idx_set:
            skipping_count += 1
            continue

        # for encoding
        original_claim_idx2true_idx = {}
        original_claim_idx2true_idx[-1] = -1
        
        
        try:
            # if True:
            for true_claim_idx, claim_info in enumerate(app_claims):
                curr_depth = 1

                # if "cancaled", add dummy node and skip
                claim, app_num, feat_encoding, original_claim_idx, label = claim_info
                graph_level_info = [
                    app_num,
                    feat_encoding,
                    original_claim_idx,
                    label,
                    leader_claims[app_idx][true_claim_idx],
                ]
                original_claim_idx2true_idx[original_claim_idx.item()] = true_claim_idx
                if _claim_is_canceled(claim) or "":
                    claim_graph = nx.DiGraph()
                    claim_graph.add_node(
                        (app_idx, true_claim_idx, 0),
                        phrase="",
                        hasChild=False,
                        identity="",
                        root=True,
                        depth=curr_depth,
                    )
                    app_graph.append(
                        (claim_graph, graph_level_info)
                    )  # edge from root to child
                    _set_attributes(
                        claim_graph,
                        app_num=app_num,
                        feat_encoding=feat_encoding,
                        label=label,
                        original_claim_idx=original_claim_idx,
                        true_claim_idx=true_claim_idx,
                    )
                    print("a cancelled claim happened")
                    continue

                # logic for non-leader claim graph:
                # if the claim is not leader
                #   copy the refered graph and add the new node
                #   find the best matched ancestor identity
                #   if the claim refers to a non-leader claim,
                #       Loop to the ancestors of the referring node,
                #   else
                #       Find the claim with best match and append the node to the claim graph specific node
                if not leader_claims[app_idx][true_claim_idx]:
                    original_referring_claim_idx = _get_referring_claim_index(claim)
                    try:
                        referring_claim_idx = original_claim_idx2true_idx[
                            original_referring_claim_idx
                        ]
                    except:
                        referring_claim_idx = true_claim_idx - 1
                    # if it is referring itself, Kewen: or referring to a cancelled claim
                    # change the referring idx to the previous claim for now.
                    while referring_claim_idx >= true_claim_idx:
                        referring_claim_idx = true_claim_idx - 1

                    claim_graph = app_graph[referring_claim_idx][0].copy()
                    claim_graph.add_node(
                        (app_idx, true_claim_idx, 0),
                        phrase=_remove_leader_vocab(claim),
                        hasChild=False,
                        identity="",
                        root=False,
                    )  # only have one cmp; empty encoding for now; non-leader claims do not have child thus identity not needed
                    # if the refered claim is a leader, find best match component
                    if leader_claims[app_idx][referring_claim_idx]:
                        best_match_component_idx = _match_claim_to_component(
                            claim,
                            identity_map[(app_idx, referring_claim_idx)],
                            pos_tagger,
                        )  # component_idx
                    else:
                        best_match_component_idx = 0

                    ancestors_node_id_list = _find_all_ancestor_claims(
                        app_idx,
                        referring_claim_idx,
                        best_match_component_idx,
                        claim_graph,
                    )

                    (
                        best_ancester_claim_idx,
                        best_ancestor_cmp_idx,
                    ) = _find_match_claim_component(
                        claim, ancestors_node_id_list, claim_graph, pos_tagger
                    )

                    # find the depth of the ancestor and add one
                    prev_depth = claim_graph.nodes[
                        (app_idx, best_ancester_claim_idx, best_ancestor_cmp_idx)
                    ]["depth"]
                    claim_graph.nodes[(app_idx, true_claim_idx, 0)]["depth"] = (
                        prev_depth + 1
                    )

                    edge_type = _get_edge_nll(claim)
                    claim_graph.add_edge(
                        (app_idx, best_ancester_claim_idx, best_ancestor_cmp_idx),
                        (app_idx, true_claim_idx, 0),
                        edge_type=edge_type,
                    )

                # logic for leader claim graph:
                # get referred/new graphs
                # add summary and connect summary if necessary
                # add component nodes and connect them to summary
                else:
                    is_root = False
                    # check if this is the root claim for its graph
                    claim_sum, _ = _decompose_claim(claim)
                    original_referring_claim_idx = _get_referring_claim_index(claim)
                    referred_claim_idx = original_claim_idx2true_idx[
                        original_referring_claim_idx
                    ]

                    # if it is referring later claims, skip for now.
                    while referred_claim_idx >= true_claim_idx:
                        referred_claim_idx = referred_claim_idx - 1

                    if referred_claim_idx == -1:
                        claim_graph = nx.DiGraph()
                        is_root = True
                    else:
                        claim_graph = app_graph[referred_claim_idx][0].copy()
                    sum_text, cmp_list = _decompose_claim(claim)

                    # add summary
                    sum_identity = identity_map[(app_idx, true_claim_idx)][0]

                    # both new claim root and reffering claim root will be set to true
                    claim_graph.add_node(
                        (app_idx, true_claim_idx, 0),
                        phrase=_remove_leader_vocab(sum_text),
                        hasChild=True,
                        identity=sum_identity,
                        root=True,
                    )

                    # connect summary if necessary
                    if not is_root:
                        # find best ancestor identity to match
                        if leader_claims[app_idx][referred_claim_idx]:
                            best_match_component_idx = _match_claim_to_component(
                                claim,
                                identity_map[(app_idx, referred_claim_idx)],
                                pos_tagger,
                            )  # component_idx
                        else:
                            best_match_component_idx = 0

                        ancestors_node_id_list = _find_all_ancestor_claims(
                            app_idx,
                            referred_claim_idx,
                            best_match_component_idx,
                            claim_graph,
                        )

                        (
                            best_ancester_claim_idx,
                            best_ancestor_cmp_idx,
                        ) = _find_match_claim_component(
                            claim, ancestors_node_id_list, claim_graph, pos_tagger
                        )

                        prev_depth = claim_graph.nodes[
                            (app_idx, best_ancester_claim_idx, best_ancestor_cmp_idx)
                        ]["depth"]
                        claim_graph.nodes[(app_idx, true_claim_idx, 0)]["depth"] = (
                            prev_depth + 1
                        )

                        edge_type = _get_edge_ll(sum_text)
                        claim_graph.add_edge(
                            (app_idx, best_ancester_claim_idx, best_ancestor_cmp_idx),
                            (app_idx, true_claim_idx, 0),
                            edge_type=edge_type,
                        )
                    else:
                        claim_graph.nodes[(app_idx, true_claim_idx, 0)]["depth"] = 1

                    # if no components, skip
                    if cmp_list == None:
                        _set_attributes(
                            claim_graph,
                            app_num=app_num,
                            feat_encoding=feat_encoding,
                            label=label,
                            original_claim_idx=original_claim_idx,
                            true_claim_idx=true_claim_idx,
                        )
                        app_graph.append((claim_graph, graph_level_info))
                        continue

                    rest_as_nested = False
                    rest_nested_cmp = []
                    last_component_idx = -1
                    # loop through components
                    for component_idx, component in enumerate(cmp_list):
                        if rest_as_nested:

                            # temp solution, for multiple nested, conver it to double nested
                            if ":" in component:
                                component = component.replace(":", ";")
                                cmp_list = component.split(";")
                                for cmp in cmp_list:
                                    rest_nested_cmp.append(cmp)
                                continue

                            rest_nested_cmp.append(component)
                            continue

                        # if the component is nested
                        # node id become (app_idx, true_claim_idx, (parent_component_idx, nested_component_idx))
                        if _check_nested_component(component)[1]:

                            # when the rest of cmp should be treated as nested
                            if _check_nested_component(component)[0] == "semi":
                                rest_as_nested = True
                                # seperate the summary from the first nested component first
                                nested_sum, nested_heading_cmp = _decompose_claim(
                                    component
                                )
                                rest_nested_cmp.append(nested_sum)

                                # temp solution, for multiple nested, conver it to double nested
                                if ":" in nested_heading_cmp[0]:

                                    component = nested_heading_cmp[0].replace(":", ";")
                                    cmp_list = component.split(";")
                                    for cmp in cmp_list:
                                        rest_nested_cmp.append(cmp)

                                else:
                                    rest_nested_cmp.append(nested_heading_cmp[0])

                                last_component_idx = (
                                    component_idx + 1
                                )  # identity map start from summary, add 1 here
                                continue

                            # iterate through nested components
                            nested_component_list = _split_component(component)
                            for n_c_i, n_c in enumerate(nested_component_list):
                                tmp_identity = identity_map[(app_idx, true_claim_idx)][
                                    component_idx + 1
                                ][n_c_i]
                                if n_c_i == 0:
                                    claim_graph.add_node(
                                        (
                                            app_idx,
                                            true_claim_idx,
                                            (component_idx + 1, n_c_i),
                                        ),
                                        phrase=_remove_leader_vocab(n_c),
                                        hasChild=True,
                                        identity=tmp_identity,
                                        root=False,
                                    )
                                    prev_depth = claim_graph.nodes[
                                        (app_idx, true_claim_idx, 0)
                                    ]["depth"]
                                    claim_graph.nodes[
                                        (
                                            app_idx,
                                            true_claim_idx,
                                            (component_idx + 1, n_c_i),
                                        )
                                    ]["depth"] = (
                                        prev_depth + 1
                                    )

                                    edge_type =_get_edge_inclaim(
                                        claim_graph.nodes[(app_idx, true_claim_idx, 0)][
                                            "phrase"
                                        ]
                                    )
                                    claim_graph.add_edge(
                                        (app_idx, true_claim_idx, 0),
                                        (
                                            app_idx,
                                            true_claim_idx,
                                            (component_idx + 1, n_c_i),
                                        ),
                                        edge_type=edge_type,
                                    )
                                    continue
                                else:
                                    claim_graph.add_node(
                                        (
                                            app_idx,
                                            true_claim_idx,
                                            (component_idx + 1, n_c_i),
                                        ),
                                        phrase=_remove_leader_vocab(n_c),
                                        hasChild=True,
                                        identity=tmp_identity,
                                        root=False,
                                    )
                                    prev_depth = claim_graph.nodes[
                                        (
                                            app_idx,
                                            true_claim_idx,
                                            (component_idx + 1, 0),
                                        )
                                    ]["depth"]
                                    claim_graph.nodes[
                                        (
                                            app_idx,
                                            true_claim_idx,
                                            (component_idx + 1, n_c_i),
                                        )
                                    ]["depth"] = (
                                        prev_depth + 1
                                    )

                                    edge_type =_get_edge_inclaim(
                                        claim_graph.nodes[
                                            (
                                                app_idx,
                                                true_claim_idx,
                                                (component_idx + 1, 0),
                                            )
                                        ]["phrase"]
                                    )
                                    claim_graph.add_edge(
                                        (
                                            app_idx,
                                            true_claim_idx,
                                            (component_idx + 1, 0),
                                        ),
                                        (
                                            app_idx,
                                            true_claim_idx,
                                            (component_idx + 1, n_c_i),
                                        ),
                                        edge_type=edge_type,
                                    )

                            continue

                        tmp_identity = identity_map[(app_idx, true_claim_idx)][
                            component_idx + 1
                        ]
                        claim_graph.add_node(
                            (app_idx, true_claim_idx, component_idx + 1),
                            phrase=_remove_leader_vocab(component),
                            hasChild=True,
                            identity=tmp_identity,
                            root=False,
                        )  # idx starts from 1 for proceeding summary; empty encoding for now
                        prev_depth = claim_graph.nodes[(app_idx, true_claim_idx, 0)][
                            "depth"
                        ]
                        claim_graph.nodes[(app_idx, true_claim_idx, component_idx + 1)][
                            "depth"
                        ] = (prev_depth + 1)

                        edge_type =_get_edge_inclaim(
                            claim_graph.nodes[(app_idx, true_claim_idx, 0)]["phrase"]
                        )

                        claim_graph.add_edge(
                            (app_idx, true_claim_idx, 0),
                            (app_idx, true_claim_idx, component_idx + 1),
                            edge_type=edge_type,
                        )

                    # special case, when the last nested component is seperated using semicolon
                    # treat the following components as nested
                    if rest_as_nested:
                        nested_component_list = rest_nested_cmp
                        for n_c_i, n_c in enumerate(nested_component_list):
                            tmp_identity = identity_map[(app_idx, true_claim_idx)][
                                last_component_idx
                            ][n_c_i]
                            if n_c_i == 0:
                                claim_graph.add_node(
                                    (
                                        app_idx,
                                        true_claim_idx,
                                        (last_component_idx, n_c_i),
                                    ),
                                    phrase=_remove_leader_vocab(n_c),
                                    hasChild=True,
                                    identity=tmp_identity,
                                    root=False,
                                )
                                prev_depth = claim_graph.nodes[
                                    (app_idx, true_claim_idx, 0)
                                ]["depth"]
                                claim_graph.nodes[
                                    (
                                        app_idx,
                                        true_claim_idx,
                                        (last_component_idx, n_c_i),
                                    )
                                ]["depth"] = (
                                    prev_depth + 1
                                )

                                edge_type =_get_edge_inclaim(
                                    claim_graph.nodes[(app_idx, true_claim_idx, 0)][
                                        "phrase"
                                    ]
                                )
                                claim_graph.add_edge(
                                    (app_idx, true_claim_idx, 0),
                                    (
                                        app_idx,
                                        true_claim_idx,
                                        (last_component_idx, n_c_i),
                                    ),
                                    edge_type=edge_type,
                                )
                                continue
                            else:
                                claim_graph.add_node(
                                    (
                                        app_idx,
                                        true_claim_idx,
                                        (last_component_idx, n_c_i),
                                    ),
                                    phrase=_remove_leader_vocab(n_c),
                                    hasChild=True,
                                    identity=tmp_identity,
                                    root=False,
                                )
                                prev_depth = claim_graph.nodes[
                                    (app_idx, true_claim_idx, (last_component_idx, 0))
                                ]["depth"]
                                claim_graph.nodes[
                                    (
                                        app_idx,
                                        true_claim_idx,
                                        (last_component_idx, n_c_i),
                                    )
                                ]["depth"] = (
                                    prev_depth + 1
                                )

                                edge_type =_get_edge_inclaim(
                                    claim_graph.nodes[
                                        (
                                            app_idx,
                                            true_claim_idx,
                                            (last_component_idx, 0),
                                        )
                                    ]["phrase"]
                                )
                                claim_graph.add_edge(
                                    (app_idx, true_claim_idx, (last_component_idx, 0)),
                                    (
                                        app_idx,
                                        true_claim_idx,
                                        (last_component_idx, n_c_i),
                                    ),
                                    edge_type=edge_type,
                                )
                _set_attributes(
                    claim_graph,
                    app_num=app_num,
                    feat_encoding=feat_encoding,
                    label=label,
                    original_claim_idx=original_claim_idx,
                    true_claim_idx=true_claim_idx,
                )
                app_graph.append((claim_graph, graph_level_info))
            meta_graphs.append(app_graph)
        except Exception as e:
            skipping_count += 1
            continue

    return meta_graphs, skipping_count

def _worker_build_graphs(batch):
    # parser client
    pos_tagger = CoreNLPParser(
        url=PARSER_SERVER_ADD, tagtype="pos"
    )
    
    # extract information
    leader_claims_mask, apps_claims_list_raw, skipping_app_idx_set = _find_leaders(batch)
    identity_map, skipping_app_idx_set = _extract_identity(
        apps_claims_list_raw, leader_claims_mask, pos_tagger, True, skipping_app_idx_set
    )
    
    # build the graph
    graphs, skipping_count = _build_batch_graphs(
        apps_claims_list_raw,
        identity_map,
        leader_claims_mask,
        pos_tagger,
        skipping_app_idx_set,
        True,
    )

    return graphs, skipping_count


def _build_meta_graphs(apps_claims_list_raw):
  
    # Start to build graphs in parallel
    graph_loader = DataLoader(
        apps_claims_list_raw,
        collate_fn=_worker_build_graphs,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # run the parser server
    os.chdir(PARSER_DIRECTORY)
    ###
    subprocess.Popen(
        PARSER_SERVER_COMMAND, shell=True, stdout=subprocess.DEVNULL
    )
    # wait for the parser server to start
    time.sleep(5)

    meta_graphs = []
    total_skipping = 0
    
    with tqdm(
        total=(len(apps_claims_list_raw)) / BATCH_SIZE - 1,
        desc="Building Graphs",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for step, (batch_graphs, skipping_count) in enumerate(graph_loader):
            pbar.update(1)
            total_skipping += skipping_count
            for i in batch_graphs:
                meta_graphs.append(i)

    return meta_graphs, total_skipping

def _encode_prep_phrase(app_idx):
    app_ids = []
    app_phrases = []
    app_graph = []
    for claim_idx, claim_info in enumerate(META_GRAPHS[app_idx]):
        claim_graph = claim_info[0]
        # [(id,phrase)...]
        claim_id_phrase_list = list(claim_graph.nodes(data="phrase"))
        claim_ids = [cip[0] for cip in claim_id_phrase_list]
        claim_phrases = [cip[1] for cip in claim_id_phrase_list]

        # change ids to meta space
        claim_new_ids = []
        for cid in claim_ids:
            if isinstance(cid[2], tuple):
                claim_new_ids.append((app_idx, cid[1], cid[2][0], cid[2][1]))
            else:
                claim_new_ids.append((app_idx, cid[1], cid[2]))
        claim_ids_mapping = dict(zip(claim_ids, claim_new_ids))
        claim_graph_new = nx.relabel_nodes(claim_graph, claim_ids_mapping, copy=True)
        # copy over attributes
        _set_attributes(
            claim_graph_new,
            app_num=claim_graph.app_num,
            feat_encoding=claim_graph.feat_encoding,
            label=claim_graph.label,
            original_claim_idx=claim_graph.original_claim_idx,
            true_claim_idx=claim_graph.true_claim_idx,
        )

        claim_graph_new_pair = (claim_graph_new, claim_info[1])

        app_ids = app_ids + claim_new_ids
        app_phrases = app_phrases + claim_phrases
        app_graph.append(claim_graph_new_pair)

    return ([[app_ids[i], app_phrases[i]] for i in range(len(app_ids))], app_graph)

def _encode_graph_nodes(meta_graphs, encoder):
    global META_GRAPHS
    # prepare phrases
    print("=====> Start to prepare phrases")
    meta_ids = []  # [(app_idx,claim_idx,cmp_idx)...]
    meta_phrases = []  # [phrase1, ...]

    meta_id_phrases = []
    META_GRAPHS = meta_graphs
    index_list = range(len(meta_graphs))
    pool = Pool(processes=NUM_WORKERS)
    results = tqdm(pool.map(_encode_prep_phrase, index_list, chunksize=32))
    meta_id_phrases_graph = np.array(list(results))
    meta_graphs_new = meta_id_phrases_graph[:, 1]
    meta_id_phrases = meta_id_phrases_graph[:, 0]
    meta_id_phrases = np.concatenate([np.array(i) for i in meta_id_phrases])

    meta_ids = meta_id_phrases[:, 0]
    meta_phrases = meta_id_phrases[:, 1]
    pool.close()

    # start to encode
    print("=====> Start to encoding phrases")
    print(datetime.now().time())
    pool = encoder.start_multi_process_pool(
        CUDA_LIST
    )
    meta_encoding = encoder.encode_multi_process(
        meta_phrases, pool, batch_size=BATCH_SIZE * 2
    )
    print(datetime.now().time())
    print("=====> Start to add encoding back to graph")
    # add encoding back to graphs
    meta_id_encoding_map = dict(zip(meta_ids, meta_encoding))
    for app_idx, app_graphs in enumerate(tqdm(meta_graphs_new)):
        for claim_idx, claim_info in enumerate(app_graphs):
            claim_graph = claim_info[0]
            _set_node_attributes(claim_graph, meta_id_encoding_map, "encoding")
    encoder.stop_multi_process_pool(pool)

    return meta_graphs_new

def _get_edge_inclaim(claim):
    for _, (k, v) in enumerate(INNER_RELATIONS.items()):
        if re.search(v, claim):
            return k
    return INC_DEFUALT_RELA

def _get_edge_nll(claim):
    for _, (k, v) in enumerate(CROSS_RELATIONS.items()):
        if re.search(v, claim):
            return k
    return NLL_DEFUALT_RELA

def _get_edge_ll(claim):
    return LL_DEFUALT_RELA

def _set_node_attributes(G, attribute_mapping, attribute):

    for node in G.nodes():
        G._node[node][attribute] = attribute_mapping[node]

def _claim_is_canceled(claim: str):
    for ck in CANCEL_INDICATOR:
        if ck in claim:
            return True

    return False

def _load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)
    return param

def _remove_stopwords(s):
    return " ".join(w for w in s.split() if w not in STOPWORDS)

def connLoadDb(db_pr, etl_pr):
    db_settings = _load_params(db_pr)
    db = pymongo.MongoClient(**db_settings)
    cleaning_settings = _load_params(etl_pr)

    set_name = cleaning_settings["set_name"]
    collection = db["utility_patents_full"][set_name]
    return collection.find(cleaning_settings[set_name])

def _preprocess_raw_claim(claim: str):

    if "(canceled)" in claim and "-" in claim:
        digit_list = re.findall("\d+", claim)
        sc = int(digit_list[0])
        ec = int(digit_list[1])
        claims = []
        for i in range(sc, ec + 1):
            claims.append("{}. (canceled)".format(i))
        return claims

    # some edge cases
    if "NO:" in claim:
        claim = claim.replace("NO:", "NO")
    if "NOs:" in claim:
        claim = claim.replace("NOs:", "NOs")

    if "ratio" in claim and ";" not in claim:
        claim = claim.replace(":", " to ")

    if ":" in claim:
        claim_reverse = claim[::-1]
        if claim_reverse.strip().index(":") == 0:
            claim = claim.replace(":", "")

    if ":" not in claim:
        return claim

    claim_list = list(claim)
    for idx, ch in enumerate(claim_list):
        if ch == ":":
            if idx != 0 and idx != len(claim) - 1:
                if claim_list[idx - 1].isdigit() and claim_list[idx + 1].isdigit():
                    claim_list[idx] = "&"

    return "".join(claim_list).replace("&", " to ")

# use the fist digit as the referring claim number
def _get_referring_claim_index(claim: string):
    try:
        return int(re.findall(r"claim (\d+)", claim)[0])
    except:
        return -1

# find the number of words overlap between the claim and the given identity
# after remove stop words
def _overlap_string_length(claim, identity):
    count = 0
    claim_no_stops = _remove_stopwords(claim)

    claim_no_stops = "".join([x for x in claim_no_stops if not x in PUNCT])
    identity_words_set = set(_remove_stopwords(identity).split())
    for w_c in claim_no_stops.split():
        if w_c in identity_words_set:
            count += 1
    return count

# replace the leader words to semicolon
def _add_colon_leader_vocab(claim: string) -> str:
    replace = False
    if "NO:" in claim:
        claim = claim.replace("NO:", "NO")

    for leader_term in LEADER_CONDITION_VOCAB:
        if leader_term in claim:
            # only replace comprising if colon was not in the claim before
            if ":" not in claim:
                # $ as a mark
                new_claim = claim.replace(leader_term, leader_term + " $$$$$ :", 1)
                replace = True

    if replace:
        return new_claim
    else:
        return claim

# replace the leader words to semicolon
def _remove_leader_vocab(phrase: string) -> str:
    if "$$$$$" in phrase:
        phrase_list = phrase.split()
        mark_index = phrase_list.index("$$$$$")
        # remove mark and keyword
        new_phrase = " ".join(
            phrase_list[: mark_index - 1] + phrase_list[mark_index + 1 :]
        )
        return new_phrase
    return phrase

def _has_leader_vocab(claim: string) -> bool:
    # check leader vocabulary
    for leader_term in LEADER_CONDITION_VOCAB:
        if leader_term in claim:
            return True
    return False

def _check_leader(claim: string) -> bool:
    # check reference
    # if no reference, then is leader

    claim_sum, _ = _decompose_claim(claim)
    referred_claim_idx = _get_referring_claim_index(claim_sum)
    if referred_claim_idx == -1 and "cancel" not in claim:
        return True

    for indicator in NESTED_NUMERICAL_INDICATOR:
        if indicator in claim:
            return True

    if claim.count(";") >= 2:
        return True

    matched = re.match("[^:;]*:[^:;]*(;[^:;]+)*", claim)
    return bool(matched)

def _check_nested_component(claim: string):
    # check if there are numerical labels
    for indicator in NESTED_NUMERICAL_INDICATOR:

        if indicator in claim and NESTED_INDICATOR_CAMP[indicator] in claim:
            return ("ok", True)

    # check if there is a semi colon
    # colon will make the followoing components became nested
    if ":" in claim:
        if (
            "ratio" in claim
            and claim[claim.index(":") - 1].isdigit()
            and claim[claim.index(":") + 1].isdigit()
        ):
            return ("ok", False)
        if claim.index(":") == len(claim) - 1:
            return ("ok", False)
        return ("semi", True)

    return ("ok", False)

def _split_component(claim: string):
    claim = "".join([c for c in claim if c not in {",", "."}])
    result_list = []
    claim = re.sub(
        "([\(\[])*[1|2|3|4|5|i|ii|iii|iv|v|vi|a|b|c|d|e|f|g|h]*?([\)\]])",
        "\g<1>\g<2>",
        claim,
    )

    if ":" in claim:
        result_list.append(claim.split(":")[0])
        rest = claim.split(":")[1].strip()
    else:
        rest = claim.strip()

    if ("()") in rest:
        for cmp in rest.split("()"):
            if cmp.strip() == "":
                continue
            result_list.append(cmp)
    else:
        for cmp in rest.split(")"):
            if cmp.strip() == "":
                continue
            result_list.append(cmp)

    return result_list

def _decompose_claim(claim: string):
    if ":" not in claim:
        return (claim, None)
    claim_sum = claim.split(":", 1)[0]
    components = claim.split(":", 1)[1].split(";", -1)
    components = [c for c in components if c.strip() != ""]
    if len(components) == 0:
        return (claim, None)
    return (claim_sum, components)

def _trim_component(cmp_list):
    cmp = " ".join(cmp_list)
    new_cmp = re.sub(
        "([\(\[])*[1|2|3|4|5|i|ii|iii|iv|v|vi|a|b|c|d|e|f|g|h]*?([\)\]])",
        "\g<1>\g<2>",
        cmp,
    ).strip()
    new_cmp = "".join([x for x in new_cmp if (not x in PUNCT) and (not x.isdigit())])
    new_cmp_list = [x for x in new_cmp.split() if x]
    if len(new_cmp_list) == 0:
        new_cmp_list.append("NA")
    return new_cmp_list

def _find_leaders(apps_claims_list):
    leader_claims = []  # 1 for leader claim, 0 for non-leader claim
    skipping_app_idx = []

    for app_idx, claims in enumerate(apps_claims_list):
        try:
            curr_leader_node_indeicator = []

            for idx, claim in enumerate(claims):
                apps_claims_list[app_idx][idx][0] = _preprocess_raw_claim(claim[0])
                claim = apps_claims_list[app_idx][idx][0]

                # check if the current claim is a leader
                if _has_leader_vocab(claim):
                    apps_claims_list[app_idx][idx][0] = _add_colon_leader_vocab(claim)
                    claim = apps_claims_list[app_idx][idx][0]
                curr_leader_node_indeicator.append(_check_leader(claim))

            leader_claims.append(curr_leader_node_indeicator)
        except:
            skipping_app_idx.append(app_idx)

    return leader_claims, apps_claims_list, set(skipping_app_idx)

# return the identity of the input tags
def _get_identity(tree):
    for subtree in tree:
        if type(subtree) == Tree and (
            subtree.label() == "identity_NN" or subtree.label() == "identiy_VBG"
        ):
            identity = ""
            for word in subtree:
                identity += word[0] + " "
            return subtree.label(), identity.strip()

    return "NA", ""

def _extract_identity(
    apps_claims_list, leader_claims, pos_tagger, mute_tqdm, skipping_app_input_set
):
    # use stanford parser. better in our case. need to run the server first

    identity_map = {}
    skipping_app_idx = []

    for app_idx, claims in enumerate(tqdm(apps_claims_list, disable=mute_tqdm)):
        if app_idx in skipping_app_input_set:
            skipping_app_idx.append(app_idx)
            continue
        try:
            for claim_idx, claim in enumerate(claims):
                claim = claim[0]
                # general logic
                # check if is leader claim. Non-leader claim does not have identity
                # get the identity of the sumamry
                # get the identities of the component, and handle the nested component

                # check if leader node
                if not leader_claims[app_idx][claim_idx]:
                    identity_map[
                        (app_idx, claim_idx)
                    ] = ""  # non_leader node does not have identity. claim-identity match
                    continue

                # identity list format:
                # [smr idt, cmp#1 idt, cmp#2 idt, ... , [nested cmp #1, nested cmp #2...],...]
                identity_list = []
                claim_sum, claim_cmp_list = _decompose_claim(claim)
                if claim_cmp_list == None:
                    claim_cmp_list = []

                # get the idenity of the summary
                summary_slplitted = claim_sum.split(" ")
                summary_slplitted = [word for word in summary_slplitted if word]
                tags = list(pos_tagger.tag(summary_slplitted))
                output = IDENTITY_PARSER.parse(tags)

                identity_list.append(_get_identity(output))

                # prepare for end nesting
                rest_as_nested = False
                rest_as_nested_components = []

                # get identity of the components
                for idx, component in enumerate(claim_cmp_list):
                    if rest_as_nested:

                        # temp solution, for multiple nested, conver it to double nested
                        if ":" in component:
                            component = component.replace(":", ";")
                            cmp_list = component.split(";")
                            for cmp in cmp_list:
                                rest_as_nested_components.append(cmp)
                            continue

                        rest_as_nested_components.append(component)
                        continue

                    # check if the component is nested
                    check_nested = _check_nested_component(component)

                    # handle nested compoenent
                    if check_nested[1]:
                        nested_identity = []

                        if check_nested[0] == "ok":
                            component = component.strip()
                            splited_component_list = _split_component(component)
                            for idx_n, nested_component in enumerate(
                                splited_component_list
                            ):
                                component_splitted = nested_component.strip().split(" ")
                                component_splitted = [
                                    word for word in component_splitted if word
                                ]
                                component_splitted = _trim_component(component_splitted)
                                tags = list(pos_tagger.tag(component_splitted))
                                output = IDENTITY_PARSER.parse(tags)

                                nested_identity.append(_get_identity(output))

                            identity_list.append(nested_identity)
                            continue
                        else:
                            rest_as_nested_summary = component.split(":")[0]

                            rest_as_nested_components += component.split(":")
                            rest_as_nested = True
                            continue
                    if component.strip() != "":
                        component_splitted = component.strip().split(" ")
                        component_splitted = [
                            word for word in component_splitted if word
                        ]
                        component_splitted = _trim_component(component_splitted)
                        tags = list(pos_tagger.tag(component_splitted))
                        output = IDENTITY_PARSER.parse(tags)
                        identity_list.append(_get_identity(output))
                    else:
                        identity_list.append("")

                if rest_as_nested == True:
                    for idx_rf, cmp in enumerate(rest_as_nested_components):
                        component_splitted = cmp.split(" ")
                        component_splitted = [
                            word for word in component_splitted if word
                        ]
                        component_splitted = _trim_component(component_splitted)
                        tags = list(pos_tagger.tag(component_splitted))
                        output = IDENTITY_PARSER.parse(tags)
                        nested_identity.append(_get_identity(output))
                    identity_list.append(nested_identity)

                identity_map[(app_idx, claim_idx)] = identity_list
        except:
            skipping_app_idx.append(app_idx)

    return identity_map, set(skipping_app_idx)

# cell of some helper functions to construct the graphs
def _match_claim_to_component(claim: string, identity_list, pos_tagger):
    # general logic:
    # if the claim has the claim_toggle keyword
    #   find the identity after the keyword, and implement identity-identity matching
    # if the claim does not have the keyword
    #   perform partial_claim-identity matching
    #   partial claim is the part of claim before referring

    # should not consider after-colon part
    colon_idx = claim.find(":")
    claim_check = claim[:colon_idx] if colon_idx != -1 else claim
    i2i = False
    tw = ""
    for toggle_word in CLAIM_MATCHING_TOGGLE:
        if toggle_word in claim_check:
            i2i = True
            tw = toggle_word

    # if the claim has the claim_toggle
    if i2i:
        # find the identity of the latter parts
        shift = 1
        later_part_claim = claim.split(tw)[shift]
        # edge case of "where where"
        while later_part_claim.strip() == "":
            shift += 1
            later_part_claim = claim.split(tw)[shift]
        tags = list(pos_tagger.tag(later_part_claim.split()))
        output = IDENTITY_PARSER.parse(tags)
        claim_identity = _get_identity(output)
        return _match_claim_to_component_helper(claim_identity[1], identity_list)
    # if the claim does not have the toggle word
    else:
        if claim[0].isdigit():
            claim = claim[1:]
        initial_part_claim = claim[
            : re.search(r"\d", claim).start()
        ]  # part of claim before referring
        return _match_claim_to_component_helper(initial_part_claim, identity_list)

# return the identity position with the longest substring overlap
def _match_claim_to_component_helper(claim: string, identity_list):
    max_overlap_length = -1
    target_identity_idex = -1
    for claim_idx, identity_pair in enumerate(identity_list):
        if isinstance(identity_pair, list):
            for nested_idx, (nested_identity_type, nested_identity) in enumerate(
                identity_pair
            ):
                over_lap_length = _overlap_string_length(claim, nested_identity)
                if over_lap_length > max_overlap_length:
                    max_overlap_length = over_lap_length
                    target_identity_idex = (claim_idx, nested_idx)

        else:
            # if identity is empty,
            # means the direct parent (refered node) is not a leader
            # just skip
            if len(identity_pair) == 0:
                continue
            identity_type, identity = identity_pair
            over_lap_length = _overlap_string_length(claim, identity)
            if over_lap_length > max_overlap_length:
                max_overlap_length = over_lap_length
                target_identity_idex = claim_idx
    return target_identity_idex

def _find_all_ancestor_claims(
    app_index: int, referred_claim_idx: int, matched_cmp_idx: int, app_graph: nx.DiGraph
):
    ancestor_list = []
    ancestor_list.append(
        (app_index, referred_claim_idx, matched_cmp_idx)
    )  # reffered node should be added as first parent
    parent_list = list(
        app_graph.predecessors((app_index, referred_claim_idx, matched_cmp_idx))
    )

    while len(parent_list) != 0:
        ancestor_list.append(parent_list[0])
        parent_list = list(app_graph.predecessors(parent_list[0]))

    return ancestor_list\

def _find_match_claim_component(claim, ancestors_list, graph, pos_tagger):

    # find all ancestor identities
    ancestor_identity_list = []
    for ancestor_node in ancestors_list:
        ancestor_identity = graph.nodes[ancestor_node]["identity"]
        ancestor_identity_list.append(ancestor_identity)

    # find the best match ancestor
    matched_ancestor_id = _match_claim_to_component(claim, ancestor_identity_list, pos_tagger)
    return (
        ancestors_list[matched_ancestor_id][1],
        ancestors_list[matched_ancestor_id][2],
    )

def _set_attributes(target, **kwargs):
    for k, v in kwargs.items():
        setattr(target, k, v)