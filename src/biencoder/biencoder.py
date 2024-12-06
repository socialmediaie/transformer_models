import json
from dataclasses import dataclass
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from biencoder_utils import flatten_dict
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from scipy.special import softmax
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.utils import ModelOutput


def flatten_dict(d, prefix=""):
    output = dict()
    for k, v in d.items():
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


def cls_token(embeddings, weights):
    return embeddings[:, 0]


def get_normalized_pooling_fn(pooling_function, temperature=1.0):
    def _pooling_function(embeddings, *args, **kwargs):
        embeddings = pooling_function(embeddings, *args, **kwargs)
        embeddings = (
            F.normalize(embeddings, p=2.0, dim=-1, eps=1e-12, out=None) / temperature
        )
        return embeddings

    return _pooling_function


def normalize_embeddings_with_temp(embeddings, temperature=1.0):
    return F.normalize(embeddings, p=2.0, dim=-1, eps=1e-12, out=None) / temperature


POOLING_FUNCTION_MAP = dict(weighted_mean=weighted_mean, cls_token=cls_token)


def multi_layer_multi_dim_embeddings(
    outputs,
    weights,
    layers=[3, 6],
    dims=[32, 64, 128, 256],
    pooling_function=weighted_mean,
):
    embeddings = []
    for l in layers:
        layer_emb = outputs.hidden_states[l]
        for d in dims:
            # emb = pooling_function(layer_emb[:, :, :d], weights)  # Use this line.
            emb = weighted_mean(layer_emb[:, :, :d], weights)  # Remove this line.
            embeddings.append(emb)
    return embeddings


def cosent_loss(logit_rel, labels):
    return torch.log(
        1
        + (
            torch.exp(logit_rel.view(-1, 1) - logit_rel.view(1, -1))
            * (labels.view(-1, 1) > labels.view(1, -1))
        ).sum()
    )


@dataclass
class BiEncoderOutput(ModelOutput):
    """BiEncoderOutput"""

    loss: torch.Tensor | None = None
    logits: torch.FloatTensor = None


class MatryoshkaLinear(nn.Linear):
    """A module which computes Linear response between two tensors of different dimensions.
    It uses a weight tensor which is of size LxD, where
    D = max possible dimension of either of the tensors
    L = Number of labels
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = min(x.shape[-1], self.weight.shape[-1])
        return nn.functional.linear(x[..., :d], self.weight[:, :d], self.bias)


class MatryoshkaBilinear(nn.Bilinear):
    """A module which computes bilinear response between two tensors of different dimensions.
    It uses a weight tensor which is of size LxDxD, where
    D = max possible dimension of either of the tensors
    L = Number of labels
    """

    def __init__(self, *args, symmetric=False, anti_symmetric=False, **kwargs):
        super(MatryoshkaBilinear, self).__init__(*args, **kwargs)
        self.symmetric = symmetric
        self.anti_symmetric = anti_symmetric
        if self.symmetric or self.anti_symmetric:
            assert (
                self.weight.shape[1] == self.weight.shape[2]
            ), f"{self.weight.shape[1]=} != {self.weight.shape[2]=} when {self.symmetric=}, {self.anti_symmetric=}"

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        d1 = min(input1.shape[-1], self.weight.shape[1])
        d2 = min(input2.shape[-1], self.weight.shape[2])
        weight = self.weight
        if self.symmetric:
            weight = 0.5 * (weight + weight.transpose(1, 2))
        elif self.anti_symmetric:
            weight = 0.5 * (weight - weight.transpose(1, 2))
        return nn.functional.bilinear(
            input1[..., :d1], input2[..., :d2], weight[:, :d1, :d2], self.bias
        )


class BiEncoder(nn.Module):
    def __init__(self, encoder_model, num_classes=2, bilinear_bias=True):
        super(BiEncoder, self).__init__()
        self.encoder = encoder_model
        hidden_size = self.encoder.config.hidden_size
        self.bilinear = nn.Bilinear(
            hidden_size, hidden_size, num_classes, bias=bilinear_bias
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input1_encoding, input2_encoding, labels=None):
        input1_outputs = self.encoder(**input1_encoding)
        input2_outputs = self.encoder(**input2_encoding)

        # Get embeddings
        input1_embeddings = input1_outputs.last_hidden_state.mean(dim=1)
        input2_embeddings = input2_outputs.last_hidden_state.mean(dim=1)

        input1_embeddings = weighted_mean(
            input1_outputs.last_hidden_state,
            input1_encoding["attention_mask"].unsqueeze(-1),
        )
        input2_embeddings = weighted_mean(
            input2_outputs.last_hidden_state,
            input2_encoding["attention_mask"].unsqueeze(-1),
        )

        # Get hidden size

        logits = self.bilinear(input1_embeddings, input2_embeddings)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return BiEncoderOutput(logits=logits, loss=loss)


class BiEncoderForClassification(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config, bilinear_bias=True):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config._name_or_path, config=config)
        hidden_size = self.encoder.config.hidden_size
        self.bilinear = nn.Bilinear(
            hidden_size, hidden_size, config.num_labels, bias=bilinear_bias
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input1_encoding, input2_encoding, labels=None):
        input1_outputs = self.encoder(**input1_encoding)
        input2_outputs = self.encoder(**input2_encoding)

        # Get embeddings
        # input1_embeddings = input1_outputs.last_hidden_state.mean(dim=1)
        # input2_embeddings = input2_outputs.last_hidden_state.mean(dim=1)

        input1_embeddings = weighted_mean(
            input1_outputs.last_hidden_state,
            input1_encoding["attention_mask"].unsqueeze(-1),
        )
        input2_embeddings = weighted_mean(
            input2_outputs.last_hidden_state,
            input2_encoding["attention_mask"].unsqueeze(-1),
        )

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

    loss: torch.Tensor | None = None
    logits: torch.FloatTensor = None
    multi_logits: list[torch.FloatTensor] = None
    multi_losses: list[torch.FloatTensor] = None


class BiEncoderForClassificationMultiLayerMultiDim(PreTrainedModel):
    config_class = AutoConfig

    def __init__(
        self,
        config,
        multi_layers=[3, 6],
        multi_dims=[32, 64, 128, 256],
        kldiv_loss=False,
        bilinear_bias=True,
        shared_bilinear=False,
        matryoshka_bilinear=False,
        use_cosent_loss=False,
        pooling_function_type="weighted_mean",
        normalize_temperature=None,
        use_prefix=None,
        **kwargs,
    ):
        print(f"Before Setting super config. {multi_layers=}, {multi_dims=}, {kwargs=}")
        super().__init__(config, **kwargs)
        print(f"After Setting super config. {config._name_or_path=}, {kwargs=}")
        self.encoder = AutoModel.from_pretrained(
            config._name_or_path,
            config=config,
            # trust_remote_code=True,
            **kwargs,
        )
        print(f"After loading self.encoder. {config._name_or_path=}")
        hidden_size = self.encoder.config.hidden_size
        self.hidden_size = hidden_size
        self.multi_layers = multi_layers
        self.multi_dims = multi_dims
        self.kldiv_loss = kldiv_loss
        self.bilinear_bias = bilinear_bias
        self.shared_bilinear = shared_bilinear
        self.matryoshka_bilinear = matryoshka_bilinear
        self.use_cosent_loss = use_cosent_loss
        self.pooling_function_type = pooling_function_type
        self._pooling_function = POOLING_FUNCTION_MAP[self.pooling_function_type]
        self.normalize_temperature = normalize_temperature
        if self.normalize_temperature is not None:
            self._pooling_function = get_normalized_pooling_fn(
                self._pooling_function, temperature=1.0
            )
        self.use_prefix = use_prefix
        if self.matryoshka_bilinear:
            self.bilinear = MatryoshkaBilinear(
                hidden_size, hidden_size, config.num_labels, bias=self.bilinear_bias
            )
        else:
            self.bilinear = nn.Bilinear(
                hidden_size, hidden_size, config.num_labels, bias=self.bilinear_bias
            )

        self.setup_multi_layer_multi_dim_modules()
        self.loss_fn = nn.CrossEntropyLoss()
        self.kldiv_loss_fn = nn.KLDivLoss(log_target=True)
        self.post_init()

    def enumerate_layer_dim(self):
        i = 0
        for l in self.multi_layers:
            for d in self.multi_dims:
                yield (i, l, d)
                i += 1

    def setup_multi_layer_multi_dim_modules(self):
        modules = []
        shared_modules = {}
        for i, l, d in self.enumerate_layer_dim():
            if self.shared_bilinear and d in shared_modules:
                m = shared_modules[d]
            else:
                if self.matryoshka_bilinear:
                    m = self.bilinear
                else:
                    m = nn.Bilinear(
                        d, d, self.config.num_labels, bias=self.bilinear_bias
                    )
                shared_modules[d] = m
            modules.append(m)
        self.multi_layer_multi_dim_modules = nn.ModuleList(modules)

    def compute_multi_layer_multi_dim_outputs(
        self,
        input1_multi_embeddings,
        input2_multi_embeddings,
        final_logits=None,
        labels=None,
    ):
        multi_logits = []
        multi_losses = []
        final_log_softmax = None
        if final_logits is not None:
            final_log_softmax = F.log_softmax(final_logits, dim=-1)
        for module, input1_embeddings, input2_embeddings in zip(
            self.multi_layer_multi_dim_modules,
            input1_multi_embeddings,
            input2_multi_embeddings,
        ):
            logits = module(input1_embeddings, input2_embeddings)
            logit_log_softmax = F.log_softmax(logits, dim=-1)
            loss = 0.0
            if self.kldiv_loss and final_log_softmax is not None:
                loss += self.kldiv_loss_fn(logit_log_softmax, final_log_softmax)
            if labels is not None:
                loss += self.loss_fn(logits, labels)
                if self.use_cosent_loss:
                    # Assume logits[:, 0] is related to label = 0
                    # -logits[:, 0] will now be is relevant label
                    loss += cosent_loss(-logits[:, 0], labels)
            multi_logits.append(logits)
            multi_losses.append(loss)
        return multi_logits, multi_losses

    def embed(self, input_encoding, **kwargs):
        input_outputs = self.encoder(
            **input_encoding, output_hidden_states=True, **kwargs
        )
        input_weights = input_encoding["attention_mask"].unsqueeze(-1)
        return input_outputs, input_weights

    def matryoshka_cross_layer_forward(
        self,
        input1_encoding,
        input2_encoding,
        labels=None,
        dims=None,
        input1_layer_indices=None,
        input2_layer_indices=None,
        **kwargs,
    ):
        if not self.matryoshka_bilinear:
            raise NotImplementedError(
                f"matryoshka_cross_layer_forward is not implemented if {self.matryoshka_bilinear=}"
            )
        if dims is None:
            dims = list(
                sorted(set(self.multi_dims + [self.encoder.config.hidden_size]))
            )
        if input1_layer_indices is None:
            input1_layer_indices = list(
                sorted(set(self.multi_layers + [self.encoder.config.num_hidden_layers]))
            )
        if input2_layer_indices is None:
            input2_layer_indices = list(
                sorted(set(self.multi_layers + [self.encoder.config.num_hidden_layers]))
            )

        input1_outputs, input1_weights = self.embed(input1_encoding, **kwargs)
        input2_outputs, input2_weights = self.embed(input2_encoding, **kwargs)

        output_logits = dict()
        output_loss = dict()
        for l1 in input1_layer_indices:
            for l2 in input2_layer_indices:
                for d in dims:
                    key = (l1, l2, d)
                    input1_embeddings = self._pooling_function(
                        input1_outputs.hidden_states[l1][:, :, :d], input1_weights
                    )
                    input2_embeddings = self._pooling_function(
                        input2_outputs.hidden_states[l2][:, :, :d], input2_weights
                    )
                    logits = self.bilinear(input1_embeddings, input2_embeddings)
                    loss = self.loss_fn(logits, labels)
                    output_logits[key] = logits
                    output_loss[key] = loss
        return output_logits, output_loss

    def forward(self, input1_encoding, input2_encoding, labels=None, **kwargs):
        input1_outputs, input1_weights = self.embed(input1_encoding, **kwargs)
        input2_outputs, input2_weights = self.embed(input2_encoding, **kwargs)

        # input1_outputs = self.encoder(**input1_encoding, output_hidden_states=True, **kwargs)
        # input2_outputs = self.encoder(**input2_encoding, output_hidden_states=True, **kwargs)

        # # Get embeddings
        # input1_weights = input1_encoding["attention_mask"].unsqueeze(-1)
        # input2_weights = input2_encoding["attention_mask"].unsqueeze(-1)
        input1_embeddings = self._pooling_function(
            input1_outputs.last_hidden_state, input1_weights
        )
        input2_embeddings = self._pooling_function(
            input2_outputs.last_hidden_state, input2_weights
        )
        logits = self.bilinear(input1_embeddings, input2_embeddings)

        # Get multi embeddings
        input1_multi_embeddings = multi_layer_multi_dim_embeddings(
            input1_outputs,
            input1_weights,
            layers=self.multi_layers,
            dims=self.multi_dims,
            pooling_function=self._pooling_function,
        )
        input2_multi_embeddings = multi_layer_multi_dim_embeddings(
            input2_outputs,
            input2_weights,
            layers=self.multi_layers,
            dims=self.multi_dims,
            pooling_function=self._pooling_function,
        )

        multi_logits, multi_losses = self.compute_multi_layer_multi_dim_outputs(
            input1_multi_embeddings,
            input2_multi_embeddings,
            final_logits=logits,
            labels=labels,
        )
        # Append overall logits in the end to maintain index with modules
        multi_logits.append(logits)

        loss = None
        if labels is not None:
            # Pre-agg
            # multi_losses = [(sum(multi_losses) / len(multi_losses))]
            loss = self.loss_fn(logits, labels)
            # Append overall logits in the end to maintain index with modules
            multi_losses.append(loss)
            if self.use_cosent_loss:
                # Assume logits[:, 0] is related to label = 0
                # -logits[:, 0] will now be is relevant label
                loss = cosent_loss(-logits[:, 0], labels)
                multi_losses.append(loss)
            loss = sum(multi_losses) / len(multi_losses)
        return BiEncoderMultiLayerMultiDimOutput(
            loss=loss,
            logits=logits,
            multi_logits=multi_logits,
            multi_losses=multi_losses,
        )

    # @torch.no_grad()
    def predict(self, input1_encoding, input2_encoding, **kwargs):
        with torch.no_grad():
            outputs = self(input1_encoding, input2_encoding, **kwargs)
        multi_logits = torch.stack(outputs.multi_logits, axis=0)
        multi_probs = F.softmax(multi_logits, -1)
        return multi_probs

    def get_biencoder_config(self):
        return dict(
            multi_layers=self.multi_layers,
            multi_dims=self.multi_dims,
            kldiv_loss=self.kldiv_loss,
            bilinear_bias=self.bilinear_bias,
            shared_bilinear=self.shared_bilinear,
            matryoshka_bilinear=self.matryoshka_bilinear,
            use_cosent_loss=self.use_cosent_loss,
            pooling_function_type=self.pooling_function_type,
            normalize_temperature=self.normalize_temperature,
            use_prefix=self.use_prefix,
        )

    def save_pretrained(self, save_directory, **kwargs):
        kwargs["safe_serialization"] = False
        super().save_pretrained(save_directory, **kwargs)
        # Save Encoder
        encoder_dir = Path(save_directory) / "encoder"
        self.encoder.save_pretrained(encoder_dir, **kwargs)
        # Save BiEncoder
        biencoder_config = self.get_biencoder_config()
        biencoder_path = Path(save_directory) / "biencoder"
        biencoder_path.mkdir(exist_ok=True)
        biencoder_config_path = biencoder_path / "config.json"
        with open(biencoder_config_path, "w+") as fp:
            json.dump(biencoder_config, fp, indent=2)
        torch.save(self.bilinear.state_dict(), biencoder_path / "bilinear.pth")
        torch.save(
            self.multi_layer_multi_dim_modules.state_dict(),
            biencoder_path / "multi_layer_multi_dim_modules.pth",
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        biencoder_path = Path(pretrained_model_name_or_path) / "biencoder"
        biencoder_config_path = biencoder_path / "config.json"
        biencoder_config = {}
        if biencoder_config_path.exists():
            print(
                f"biencoder_config_path exists. Loading biencoder_config from: {biencoder_config_path=}"
            )
            with open(biencoder_config_path) as fp:
                biencoder_config = json.load(fp)
            print(f"{biencoder_config=}")
        for k, v in biencoder_config.items():
            if k not in kwargs:
                # only update not explicitly specified fields
                kwargs[k] = v
        print(f"Using {kwargs=}")
        # if "config" not in kwargs:
        #     config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        #     print(f"{config=}")
        #     kwargs["config"] = config
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def compute_metrics(self, eval_preds):
        # metric = evaluate.load("bstrai/classification_report")
        (logits, multi_logits, multi_losses), labels = eval_preds
        # print(f"compute_metrics {len(logits)=}, {len(multi_logits)=}, {len(multi_losses)=}, {len(labels)=}")
        # print(f"{len(multi_logits[0])=}, {len(multi_losses[0])=}, {multi_logits[0].shape=}, {multi_losses[0]=}")
        predictions = np.argmax(logits, axis=-1)
        metric_dict = classification_report(
            y_pred=predictions, y_true=labels, output_dict=True
        )
        for i, l, d in self.enumerate_layer_dim():
            predictions = np.argmax(multi_logits[i], axis=-1)
            # layer_dim_metric_dict = metric.compute(predictions=predictions, references=labels)
            layer_dim_metric_dict = classification_report(
                y_pred=predictions, y_true=labels, output_dict=True
            )
            # layer_dim_metric_dict["loss"] = multi_losses[i]
            key = f"layer_{l}_dim_{d}"
            # print(f"{(i,l,d)=}\t{key=}\t{multi_losses[i]=}")
            metric_dict[key] = layer_dim_metric_dict
        return flatten_dict(metric_dict)

    def parse_predictions(self, eval_preds):
        (logits, multi_logits, multi_losses), labels = eval_preds
        all_predictions = dict()
        predictions = np.argmax(logits, axis=-1)
        probs = softmax(logits, axis=-1)
        pMax = probs.max(axis=-1)
        pRel = 1 - probs[:, 0]
        all_predictions["final_pred"] = predictions
        all_predictions["final_pMax"] = pMax
        all_predictions["final_pRel"] = pRel
        for label_idx in range(probs.shape[1]):
            all_predictions[f"final_p{label_idx}"] = probs[:, label_idx]
        for i, l, d in self.enumerate_layer_dim():
            predictions = np.argmax(multi_logits[i], axis=-1)
            probs = softmax(multi_logits[i], axis=-1)
            pMax = probs.max(axis=-1)
            pRel = 1 - probs[:, 0]
            key = f"layer_{l}_dim_{d}"
            all_predictions[f"{key}_pred"] = predictions
            all_predictions[f"{key}_pMax"] = pMax
            all_predictions[f"{key}_pRel"] = pRel
            for label_idx in range(probs.shape[1]):
                all_predictions[f"{key}_p{label_idx}"] = probs[:, label_idx]
        return all_predictions

    @classmethod
    def get_transform_example_fn(cls, use_prefix=None, col_map=None):
        if col_map is None:
            col_map = dict(
                query="query",
                document="document",
                labels="labels",
            )
        query_col = col_map["query"]
        document_col = col_map["document"]
        label_col = col_map["labels"]

        def _transform_example_with_prefix(e):
            return (
                f'{use_prefix["query"]}{e[query_col]}',
                f'{use_prefix["document"]}{e[document_col]}',
                e[label_col],
            )

        def _transform_example_without_prefix(e):
            return (e[query_col], e[document_col], e[label_col])

        if use_prefix:
            return _transform_example_with_prefix
        return _transform_example_without_prefix

    def transform_example(self, e):
        if self.use_prefix:
            return (
                f'{self.use_prefix["query"]}{e["query"]}',
                f'{self.use_prefix["document"]}{e["document"]}',
                e["labels"],
            )
        return (e["query"], e["document"], e["labels"])
