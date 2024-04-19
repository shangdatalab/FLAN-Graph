# Author: Zhaoyi Hou (Joey), Yifei Ning (Couson)
# reference: https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py#L1501

import torch.nn as nn
import torch
from transformers import (
    BertModel,
    XLNetModel,
    BertPreTrainedModel,
    XLNetPreTrainedModel,
    BertForSequenceClassification,
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaConfig,
    RobertaModel,
    RobertaPreTrainedModel,
    AutoModel
)

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_utils import SequenceSummary

class BertForPatentPrediction(BertPreTrainedModel):
    def __init__(self, config, trainer_config):
        super().__init__(config)
        self.trainer_config = trainer_config
        self.num_labels = config.num_labels
        self.app_feat_length = trainer_config["app_feat_length"]
        self.mid_dim = trainer_config["mid_dim"]
        self.bert = BertModel(config)

        # Start - Customized Part
        self.fc1 = nn.Linear(config.hidden_size, self.mid_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.classifier = nn.Linear(
            self.mid_dim + self.app_feat_length, self.num_labels
        )
        # End - Customized Part

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

        self.use_hinge_loss = trainer_config["use_hinge_loss"]
        self.device_num = trainer_config["device_num"]

        self.hinge_idx = [0] #if "hinge_idx" not in trainer_config.keys() else trainer_config["hinge_idx"]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # Start - Customized Part
        app_feats=None,  # Added application features
        # End - Customized Part
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict
        transformer_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
   
        pooled_output = transformer_outputs[1]
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.fc1(pooled_output)
        pooled_output_1 = pooled_output.clone()
        pooled_output_2 = pooled_output.clone()

        # Start - Customized Part
        if self.app_feat_length > 0:
            pooled_output_1 = torch.cat((pooled_output_1, app_feats.float()), dim=1)
        logits = self.classifier(pooled_output_1)
        logits = self.softmax(logits)
        # End - Customized Part

        if self.use_hinge_loss:
            app_feats_copy = self.manufacture_constraint(
                app_feats, self.device_num, ix_lst=self.hinge_idx
            )
            pooled_output_2 = torch.cat(
                (pooled_output_2, app_feats_copy.float()), dim=1
            )
            logits_copy = self.classifier(pooled_output_2)
            logits_copy = self.softmax(logits_copy)
            return (
                logits,
                transformer_outputs[2:],
                logits_copy,
                transformer_outputs[2:],
                pooled_output,
            )
        return (logits, transformer_outputs[2:], pooled_output)

    def manufacture_constraint(self, vec, device_num, constraint_lst=[0.5], ix_lst=[0]):
        assert len(constraint_lst) == len(ix_lst), "Input Length Mismatch"

        factor_tensor = torch.zeros_like(vec)
        factor_tensor = factor_tensor.fill_(1.0)
        for ix in ix_lst:
            index = torch.tensor([ix]).cuda(device_num)
            factor_tensor = factor_tensor.index_fill_(
                1, index, constraint_lst[ix_lst.index(ix)]
            )

        out = torch.mul(factor_tensor, vec).cuda(device_num)
        return out


#Llama2 based models
class Llama2ForPatentPrediction(LlamaPreTrainedModel):
    def __init__(self, config, trainer_config):
        super().__init__(config)
        self.trainer_config = trainer_config
        self.num_labels = config.num_labels
        self.app_feat_length = trainer_config["app_feat_length"]
        self.mid_dim = trainer_config["mid_dim"]
        self.llama = LlamaModel(config)
        self.fc1 = nn.Linear(config.hidden_size, self.mid_dim)

        # Start - Customized Part
        self.softmax = nn.LogSoftmax(dim=1)
        self.fc2 = nn.Linear(self.mid_dim + self.app_feat_length, self.num_labels)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        # End - Customized Part

        self.init_weights()

        self.use_hinge_loss = trainer_config["use_hinge_loss"]
        self.device_num = trainer_config["device_num"]

        self.hinge_idx = [0]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # Start - Customized Part
        app_feats=None,  # Added application features
        # End - Customized Part
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = False
        transformer_outputs = self.llama(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        # logits = self.fc1(hidden_states)
        if self.app_feat_length > 0:
            logits = hidden_states.clone()
            # logits = self.score(hidden_states)

            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = inputs_embeds.shape[0]

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                        logits.device
                    )
                else:
                    sequence_lengths = -1

            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths] #last sequence token tensors
        
            pooled_logits = self.fc1(pooled_logits)
            # # Start - Customized Part
            # # concat app features
            if self.app_feat_length > 0:
                pooled_logits = torch.cat((pooled_logits.half(), app_feats.half()), dim=1)

            logits_out = self.fc2(pooled_logits.half())
            logits_out = self.softmax(logits_out)

            output = (logits_out, transformer_outputs[1:], pooled_logits)
            return output
        else:    
            logits = self.score(hidden_states)

            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = inputs_embeds.shape[0]

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                        logits.device
                    )
                else:
                    sequence_lengths = -1

            #sequence lengths has the index of last token for each batch
            #pooled_logits is [batch_size, last_token]
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths] #last sequence token tensors

            logits_out = self.softmax(pooled_logits)

            output = (logits_out, transformer_outputs[1:], pooled_logits)
            return output
            
        return None      

    def manufacture_constraint(self, vec, device_num, constraint_lst=[0.5], ix_lst=[0]):
        assert len(constraint_lst) == len(ix_lst), "Input Length Mismatch"

        factor_tensor = torch.zeros_like(vec)
        factor_tensor = factor_tensor.fill_(1.0)
        for ix in ix_lst:
            index = torch.tensor([ix]).cuda(device_num)
            factor_tensor = factor_tensor.index_fill_(
                1, index, constraint_lst[ix_lst.index(ix)]
            )

        out = torch.mul(factor_tensor, vec).cuda(device_num)
        return out

# Roberta based model
class RobertaForPatentPrediction(RobertaPreTrainedModel):
    def __init__(self, config, trainer_config):
        super().__init__(config)
        self.trainer_config = trainer_config
        self.num_labels = config.num_labels
        self.app_feat_length = trainer_config["app_feat_length"]
        self.mid_dim = trainer_config["mid_dim"]
        self.roberta = RobertaModel(config=config)

        # Start - Customized Part
        self.fc1 = nn.Linear(config.hidden_size, self.mid_dim)
        self.softmax = nn.LogSoftmax(dim=1)

        self.classifier = nn.Linear(
            self.mid_dim + self.app_feat_length, self.num_labels
        )
        # End - Customized Part

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

        self.use_hinge_loss = trainer_config["use_hinge_loss"]
        self.device_num = trainer_config["device_num"]

        self.hinge_idx = [0] #if "hinge_idx" not in trainer_config.keys() else trainer_config["hinge_idx"]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # Start - Customized Part
        app_feats=None,  # Added application features
        # End - Customized Part
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict

        transformer_outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
   
        pooled_output = transformer_outputs[1]
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.fc1(pooled_output)
        pooled_output_1 = pooled_output.clone()
        pooled_output_2 = pooled_output.clone()

        # Start - Customized Part
        if self.app_feat_length > 0:
            pooled_output_1 = torch.cat((pooled_output_1, app_feats.float()), dim=1)
        logits = self.classifier(pooled_output_1)
        logits = self.softmax(logits)
        # End - Customized Part

        if self.use_hinge_loss:
            app_feats_copy = self.manufacture_constraint(
                app_feats, self.device_num, ix_lst=self.hinge_idx
            )
            pooled_output_2 = torch.cat(
                (pooled_output_2, app_feats_copy.float()), dim=1
            )
            logits_copy = self.classifier(pooled_output_2)
            logits_copy = self.softmax(logits_copy)
            return (
                logits,
                transformer_outputs[2:],
                logits_copy,
                transformer_outputs[2:],
                pooled_output,
            )
        return (logits, transformer_outputs[2:], pooled_output)

    def manufacture_constraint(self, vec, device_num, constraint_lst=[0.5], ix_lst=[0]):
        assert len(constraint_lst) == len(ix_lst), "Input Length Mismatch"

        factor_tensor = torch.zeros_like(vec)
        factor_tensor = factor_tensor.fill_(1.0)
        for ix in ix_lst:
            index = torch.tensor([ix]).cuda(device_num)

            factor_tensor = factor_tensor.index_fill_(
                1, index, constraint_lst[ix_lst.index(ix)]
            )

        out = torch.mul(factor_tensor, vec).cuda(device_num)
        return out