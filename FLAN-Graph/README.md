# FLAN Graph
This folder contains the construction process of FLAN Graph and patent approval prediction with it.

## 1. Construction
The code for constructing FLAN Graph is provided [here](/FLAN-Graph/graph_construction/).

### Processing FLAN Graph


### Saved FLAN Graph


## 2. Prediction
The code for predicting patent approval with FLAN Graph is provided [here](/FLAN-Graph/graph_prediction/).

### To train on naive graph
1. Define hyperparameters and data path in `config/config_baseline.yaml`
2. run `python ./src/run_baseline.py --config ./config/config_baseline.yaml`

### To train on FLAN graph
1. Define hyperparameters and data path in `config/config_flan.yaml`
2. run `python ./src/run_flan.py --config ./config/config_flan.yaml`