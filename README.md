# FLAN Graph for Patent Approval Prediction

Source code and dataset for our paper "[Beyond Scaling: Predicting Patent Approval with Domain-specific Fine-grained Claim Dependency Graph]()"

## 0. Overview

### Code
Our code consists of the following two parts: **(1) Scaling up SOTA with LLMs** and **(2) FLAN Graph**. 

### Data
The **PatentAP** dataset we use in the paper is on 🤗 Huggingface [[link](https://huggingface.co/datasets/shangdatalab-ucsd/PatentAP)].

## 1. Scaling up with LLMs

### 1.1 Embedding-based

The code base for embedding-based training and inference can be found at 

#### Run

Running the code base requires a config file. Some samples of the config file are present in 

A new custom config file can also be created using the below template
```
{
    "model_name": "<hugging face model path>", 
    "from_checkpoint": "0", # Continue training from checkpoint
    "mid_dim": 400, # Dimension of intermediate feature layer
    "num_labels": 2, # Number of classification lables
    "device_num": 0, # GPU number(deprecated)
    "dataloader_ready": 1, # Is data loader already present? Use 0 to continue with datat preperation
    "dataloader_name": "claim_102_llama_small_app_feats_minmax", # if data loader is ready provide its path
    "testing": 0, # is inference
    "curr_label": "claim_label_102", #Label
    "batch_size": 8, # batch size
    "max_length": 128, # Maximum token length
    "max_training_steps": 0, # maximum training steps
    "use_hinge_loss": false, # In case of bert use hinge loss
    "last_dim": 10, # last dimension size
    "lr": 7e-5, #Learning rate
    "total_epochs": 2, # Epochs
    "hinge_lambda": 1e-2, # Lamba for hinge loss
    "num_feat": ["max_score_y","max_citations", "max_other_citations", "max_article_citations", "lexical_diversity"], # Additional numerical features
    "cat_feat": ["patent_class", "applicantCitedExaminerReferenceIndicatorCount", "component", "transitional_phrase"], # Additional categorical features
    "hf_token": "hf_<token>", # hugging face token to use for restricted repositories
    "use_lora": true, # Use Lora 
    "use_8bit_adam": true # Use 8 bit adam optimizer
}

```

### Training
The training can be done using the command:
`python run.py -r -m llama2_only`

### Testing
The training can be done using the command:
`python run.py -t -m llama2_only`

#### Dependencies
- See `requirements.txt`


### 1.2 Prompting-based
The prompt templates we used are provided [here](/Scaling_up_w_LLMs/prompt-based/).

## 2. Customized FLAN Graph
The construction process and the saved results of the FLAN Graph can be found [here](/FLAN-Graph/graph_construct/).


## Citation