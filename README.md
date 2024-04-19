# FLAN Graph for Patent Approval Prediction

Source code and dataset for our paper "[Beyond Scaling: Predicting Patent Approval with Domain-specific Fine-grained Claim Dependency Graph]()"

## 0. Overview

### Code
Our code consists of the following two parts: **(1) Scaling up SOTA with LLMs** and **(2) FLAN Graph**. 

### Data
The **PatentAP** dataset we use in the paper is on 🤗 Huggingface [[link](https://huggingface.co/datasets/shangdatalab-ucsd/PatentAP)].

## 1. Scaling up with LLMs

### 1.1 Embedding-based

The code base for embedding-based training and inference can be found [here](/Scaling_w_LLMs/embedding-based/).

### 1.2 Prompting-based
The prompt templates we used are provided [here](/Scaling_w_LLMs/prompt-based/).

## 2. Customized FLAN Graph
The construction process and the saved results of the FLAN Graph can be found [here](/FLAN-Graph/graph_construct/).


## Citation