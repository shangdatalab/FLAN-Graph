# FLAN Graph
This folder contains the construction process of FLAN Graph and patent approval prediction with it.

## 1. Construction
The code for constructing FLAN Graph is provided [here](/FLAN-Graph/graph_construction/).

### Processing FLAN Graph
* Download the [StanfordCoreNLPServer](https://stanfordnlp.github.io/CoreNLP/download.html)
* Populate Global variables in the [tree_builder.py](/FLAN-Graph/graph_construction/src/tree_builder.py)
* Run `python ./src/build.py --csv_path path_to_csv_data -- graph_path path_to_save_graphs --root2child true`

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