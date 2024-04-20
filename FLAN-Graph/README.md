# FLAN Graph
This folder contains the construction process of FLAN Graph and patent approval prediction with it.

## 1. Construction
The code for constructing FLAN Graph is provided [here](/FLAN-Graph/graph_construction/).

### Processing FLAN Graph
* Specify the raw csv files for test, validation, and test in the [config](/FLAN-Graph/graph_construction/config/config_build_graph.yaml).
* Specify the output graph files for test, validation, and test in the [config](/FLAN-Graph/graph_construction/config/config_build_graph.yaml). These graph will be used for training.
* Run the [StanfordCoreNLPServer](https://stanfordnlp.github.io/CoreNLP/download.html), and populate Global variables in the [tree_builder.py](/FLAN-Graph/graph_construction/src/tree_builder.py)
* Run `python ./src/build.py --root2child true --config ./config/config_build_graph.yaml`


## Saved FLAN Graph
The constructed FLAN-Graph is saved in [here](link pending)


## 2. Prediction
The code for predicting patent approval with FLAN Graph is provided [here](/FLAN-Graph/graph_prediction/).

### To train on naive graph
1. Define hyperparameters and data path in `config/config_baseline.yaml`
2. run `python ./src/run_baseline.py --config ./config/config_baseline.yaml`

### To train on FLAN graph
1. Define hyperparameters and data path in `config/config_flan.yaml`
2. run `python ./src/run_flan.py --config ./config/config_flan.yaml`