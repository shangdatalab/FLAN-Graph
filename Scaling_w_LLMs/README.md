
# Scaling with LLMs

## 1. Embedding-based
The code is provided [here](/Scaling_w_LLMs/embedding-based/).

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

## Prompt-based
The prompt templates are provided [here](/Scaling_w_LLMs/prompt-based/template.py), covering LLaMA, Mixtral, Vicuna, and OpenAI GPT models.