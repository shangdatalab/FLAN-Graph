# Construction of FLAN Graph
## Construction Process
* Specify the raw csv files for test, validation, and test in the [config](/FLAN-Graph/graph_construction/config/config_build_graph.yaml).
* Specify the output graph files for test, validation, and test in the [config](/FLAN-Graph/graph_construction/config/config_build_graph.yaml).
* Run the [StanfordCoreNLPServer](https://stanfordnlp.github.io/CoreNLP/download.html), and populate Global variables in the [tree_builder.py](/FLAN-Graph/graph_construction/src/tree_builder.py)
* Run `python ./src/build.py --root2child true --config ./config/config_build_graph.yaml`


## Saved FLAN Graph
The constructed FLAN-Graph is saved in [here](link pending)