import json
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import numpy as np
import evaluate

from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics import classification_report


def flatten_dict(d, prefix=""):
    output = dict()
    for k,v in d.items():
        key = f"{prefix}{k}"
        # print(f"{prefix=}, {k=}, {v=}")
        if isinstance(v, dict):
            output.update(flatten_dict(v, prefix=f"{key}__"))
        else:
            output[key] = v
    # print(f"{prefix=}, {output=}")
    return output


def compute_metrics(eval_preds):
    metric = evaluate.load("bstrai/classification_report")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    metric_dict = metric.compute(predictions=predictions, references=labels)
    return flatten_dict(metric_dict)


def compute_metrics_multi(eval_preds):
    metric = evaluate.load("bstrai/classification_report")
    (logits, _, _), labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    metric_dict = metric.compute(predictions=predictions, references=labels)
    return flatten_dict(metric_dict)


def weighted_mean(embeddings, weights):
    return (embeddings * weights).sum(dim=1) / weights.sum(dim=1)

def multi_layer_multi_dim_embeddings(outputs, weights, layers=[3, 6], dims=[32, 64, 128, 256]):
    embeddings = []
    for l in layers:
        layer_emb = outputs.hidden_states[l]
        for d in dims:
            emb = weighted_mean(layer_emb[:, :, :d], weights)
            embeddings.append(emb)
    return embeddings


@dataclass
class BiEncoderOutput(ModelOutput):
    """BiEncoderOutput"""
    loss: (torch.Tensor | None) = None
    logits: torch.FloatTensor = None


class BiEncoder(nn.Module):
    def __init__(self, encoder_model, num_classes=2):
        super(BiEncoder, self).__init__()
        self.encoder = encoder_model
        hidden_size = encoder_model.config.hidden_size
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        
        
    def forward(self, input1_encoding, input2_encoding, labels=None):
        input1_outputs = self.encoder(**input1_encoding)
        input2_outputs = self.encoder(**input2_encoding)
    
        # Get embeddings
        input1_embeddings = input1_outputs.last_hidden_state.mean(dim=1)
        input2_embeddings = input2_outputs.last_hidden_state.mean(dim=1)


        input1_embeddings = weighted_mean(input1_outputs.last_hidden_state, input1_encoding["attention_mask"].unsqueeze(-1))
        input2_embeddings = weighted_mean(input2_outputs.last_hidden_state, input2_encoding["attention_mask"].unsqueeze(-1))
        
        # Get hidden size
        
        logits = self.bilinear(input1_embeddings, input2_embeddings)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return BiEncoderOutput(logits=logits, loss=loss)


class BiEncoderForClassification(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(
            config._name_or_path, config=config
        )        
        hidden_size = self.encoder.config.hidden_size
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input1_encoding, input2_encoding, labels=None):
        input1_outputs = self.encoder(**input1_encoding)
        input2_outputs = self.encoder(**input2_encoding)
    
        # Get embeddings
        # input1_embeddings = input1_outputs.last_hidden_state.mean(dim=1)
        # input2_embeddings = input2_outputs.last_hidden_state.mean(dim=1)


        input1_embeddings = weighted_mean(input1_outputs.last_hidden_state, input1_encoding["attention_mask"].unsqueeze(-1))
        input2_embeddings = weighted_mean(input2_outputs.last_hidden_state, input2_encoding["attention_mask"].unsqueeze(-1))
        
        # Get hidden size
        
        logits = self.bilinear(input1_embeddings, input2_embeddings)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return BiEncoderOutput(logits=logits, loss=loss)


    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        encoder_dir = Path(save_directory) / "encoder"
        self.encoder.save_pretrained(encoder_dir, **kwargs)

    def compute_metrics(self, eval_preds):
        metric = evaluate.load("bstrai/classification_report")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        metric_dict = metric.compute(predictions=predictions, references=labels)
        return flatten_dict(metric_dict)


@dataclass
class BiEncoderMultiLayerMultiDimOutput(ModelOutput):
    """BiEncoderOutput"""
    loss: (torch.Tensor | None) = None
    logits: torch.FloatTensor = None
    multi_logits: list[torch.FloatTensor] = None
    multi_losses: list[torch.FloatTensor] = None

class BiEncoderForClassificationMultiLayerMultiDim(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config, multi_layers=[3, 6], multi_dims=[32, 64, 128, 256]):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(
            config._name_or_path, config=config
        )        
        hidden_size = self.encoder.config.hidden_size
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, config.num_labels)
        self.multi_layers = multi_layers
        self.multi_dims = multi_dims
        self.setup_multi_layer_multi_dim_modules()
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def enumerate_layer_dim(self):
        i = 0
        for l in self.multi_layers:
            for d in self.multi_dims:
                yield (i, l, d)

    def setup_multi_layer_multi_dim_modules(self):
        modules = []
        for i, l, d in self.enumerate_layer_dim():
            m = nn.Bilinear(d, d, self.config.num_labels)
            modules.append(m)
        self.multi_layer_multi_dim_modules = nn.ModuleList(modules)


    def compute_multi_layer_multi_dim_outputs(self, input1_multi_embeddings, input2_multi_embeddings, labels=None):
        multi_logits = []
        multi_losses = []
        for module, input1_embeddings, input2_embeddings in zip(
            self.multi_layer_multi_dim_modules,
            input1_multi_embeddings,
            input2_multi_embeddings
        ):
            logits = module(input1_embeddings, input2_embeddings)
            loss = None
            if labels is not None:
                loss = self.loss_fn(logits, labels)
            multi_logits.append(logits)
            multi_losses.append(loss)
        return multi_logits, multi_losses

    def forward(self, input1_encoding, input2_encoding, labels=None):
        input1_outputs = self.encoder(**input1_encoding, output_hidden_states=True)
        input2_outputs = self.encoder(**input2_encoding, output_hidden_states=True)
    
        # Get embeddings
        input1_weights = input1_encoding["attention_mask"].unsqueeze(-1)
        input2_weights = input2_encoding["attention_mask"].unsqueeze(-1)
        input1_embeddings = weighted_mean(input1_outputs.last_hidden_state, input1_weights)
        input2_embeddings = weighted_mean(input2_outputs.last_hidden_state, input2_weights)
        logits = self.bilinear(input1_embeddings, input2_embeddings)

        # Get multi embeddings
        input1_multi_embeddings = multi_layer_multi_dim_embeddings(input1_outputs, input1_weights, layers=self.multi_layers, dims=self.multi_dims)
        input2_multi_embeddings = multi_layer_multi_dim_embeddings(input2_outputs, input2_weights, layers=self.multi_layers, dims=self.multi_dims)

        multi_logits, multi_losses = self.compute_multi_layer_multi_dim_outputs(input1_multi_embeddings, input2_multi_embeddings, labels=labels)
        # Append overall logits in the end to maintain index with modules
        multi_logits.append(logits)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            # Append overall logits in the end to maintain index with modules
            multi_losses.append(loss)
            loss = sum(multi_losses) / len(multi_losses)
        return BiEncoderMultiLayerMultiDimOutput(loss=loss, logits=logits, multi_logits=multi_logits, multi_losses=multi_losses)


    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        encoder_dir = Path(save_directory) / "encoder"
        self.encoder.save_pretrained(encoder_dir, **kwargs)

    def compute_metrics(self, eval_preds):
        metric = evaluate.load("bstrai/classification_report")
        (logits, multi_logits, multi_losses), labels = eval_preds
        print(f"compute_metrics {len(logits)=}, {len(multi_logits)=}, {len(multi_losses)=}, {len(labels)=}")
        print(f"{len(multi_logits[0])=}, {len(multi_losses[0])=}, {multi_logits[0].shape=}, {multi_losses[0]=}")
        
        predictions = np.argmax(logits, axis=-1)
        metric_dict = classification_report(y_pred=predictions, y_true=labels, output_dict=True)
        for i, l, d in self.enumerate_layer_dim():
            predictions = np.argmax(multi_logits[i], axis=-1)
            # layer_dim_metric_dict = metric.compute(predictions=predictions, references=labels)
            layer_dim_metric_dict = classification_report(y_pred=predictions, y_true=labels, output_dict=True)
            # layer_dim_metric_dict["loss"] = multi_losses[i]
            key = f"layer_{l}_dim_{d}"
            print(f"{(i,l,d)=}\t{key=}\t{multi_losses[i]=}")
            metric_dict[key] = layer_dim_metric_dict            
        return flatten_dict(metric_dict)
