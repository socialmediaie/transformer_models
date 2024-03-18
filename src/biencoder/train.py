"""Run BiEncoder Experiments

Run as:
accelerate launch train.py

"""

import json
from dataclasses import dataclass

import evaluate
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from biencoder import (BiEncoder, BiEncoderForClassification, BiEncoderOutput,
                       weighted_mean, BiEncoderForClassificationMultiLayerMultiDim, BiEncoderMultiLayerMultiDimOutput, compute_metrics)
from datasets import ClassLabel, Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModel, AutoTokenizer, Trainer, TrainingArguments,
                          get_linear_schedule_with_warmup, set_seed)
from transformers.training_args import is_accelerate_available

import os

experiment_name = "BiEncoder"

os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
print(f"Using {os.environ['MLFLOW_EXPERIMENT_NAME']=}")

DRY_RUN = False

print(f"{DRY_RUN=}")

model_name_or_path = "./models/biencoder_model"
batch_size = 64
num_train_epochs = 20

use_peft = False

if DRY_RUN:
    num_train_epochs = 1

lr = 3e-4

multi_layers=[3, 6]
multi_dims=[32, 64, 128, 256]

use_multi_model = False

print(f"{use_multi_model=}")

output_dir = "./models/biencoder_model-single"
if use_multi_model:
    output_dir = "./models/biencoder_model-multi"
dataset_path = "./data/dataset_biencoder"





# Based on: https://github.com/huggingface/peft/blob/v0.8.1/examples/sequence_classification/peft_no_lora_accelerate.py


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        # evaluation_strategy="steps",
        # eval_steps=1000,
        save_strategy="epoch",
        # save_strategy="steps",
        # save_steps=1000,
        save_total_limit=1,
        dataloader_num_workers=10,
        remove_unused_columns=False,
        num_train_epochs=3.0 if DRY_RUN else num_train_epochs,
        max_steps=2 if DRY_RUN else -1,
        # max_steps=2, # Uncomment
        push_to_hub=False,
        hub_model_id=output_dir,
        load_best_model_at_end=True,
        # label_smoothing_factor=0.1
    )

    with training_args.main_process_first():
        dataset = load_from_disk(dataset_path)
        num_classes = dataset["train"].features["labels"].num_classes
        print(dataset)
        print(f"{num_classes=}")

    def tokenized_collate_fn(examples):
        # return tokenizer.pad(examples, padding="longest", return_tensors="pt")
        queries, documents, labels = zip(
            *((e["query"], e["document"], e["labels"]) for e in examples)
        )
        query_encoding = tokenizer(
            list(queries),
            truncation=True,
            max_length=None,
            padding="longest",
            return_tensors="pt",
        )
        document_encoding = tokenizer(
            list(documents),
            truncation=True,
            max_length=None,
            padding="longest",
            return_tensors="pt",
        )
        labels = torch.tensor(list(labels))
        outputs = dict(
            input1_encoding=query_encoding,
            input2_encoding=document_encoding,
            labels=labels,
        )
        return outputs

    # Make Model
    # encoder_model = AutoModel.from_pretrained(model_name_or_path, return_dict=True, num_labels=num_classes)
    model = BiEncoderForClassification.from_pretrained(
        model_name_or_path, num_labels=num_classes
    )
    # No PEFT task type for auto model loading
    if use_peft:
        peft_config = LoraConfig(
            task_type=None, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1
        )
        model.encoder = get_peft_model(model.encoder, peft_config)
        model.encoder.print_trainable_parameters()

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    if DRY_RUN:
        train_dataset = eval_dataset

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=tokenized_collate_fn,
        train_dataset=train_dataset,
        # train_dataset=dataset["test"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=model.compute_metrics,
    )

    train_output = trainer.train()
    trainer.save_model()
    trainer.save_state()
    trainer.log_metrics("train_result", train_output.metrics)
    trainer.save_metrics("train_result", train_output.metrics)


    for split in dataset:
        print(split)
        # if split in {"train", "test"}: continue
        metrics = trainer.evaluate(dataset[split])
        trainer.save_metrics(split, metrics)

    # trainer.push_to_hub()


def main_multi():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        # evaluation_strategy="steps",
        # eval_steps=1000,
        save_strategy="epoch",
        # save_strategy="steps",
        # save_steps=1000,
        save_total_limit=1,
        dataloader_num_workers=10,
        remove_unused_columns=False,
        num_train_epochs=3.0 if DRY_RUN else num_train_epochs,
        max_steps=2 if DRY_RUN else -1,
        # max_steps=2, # Uncomment
        push_to_hub=False,
        hub_model_id=output_dir,
        load_best_model_at_end=True,
        # label_smoothing_factor=0.1
    )

    with training_args.main_process_first():
        dataset = load_from_disk(dataset_path)
        num_classes = dataset["train"].features["labels"].num_classes
        print(dataset)
        print(f"{num_classes=}")

    def tokenized_collate_fn(examples):
        # return tokenizer.pad(examples, padding="longest", return_tensors="pt")
        queries, documents, labels = zip(
            *((e["query"], e["document"], e["labels"]) for e in examples)
        )
        query_encoding = tokenizer(
            list(queries),
            truncation=True,
            max_length=None,
            padding="longest",
            return_tensors="pt",
        )
        document_encoding = tokenizer(
            list(documents),
            truncation=True,
            max_length=None,
            padding="longest",
            return_tensors="pt",
        )
        labels = torch.tensor(list(labels))
        outputs = dict(
            input1_encoding=query_encoding,
            input2_encoding=document_encoding,
            labels=labels,
        )
        return outputs

    # Make Model
    # encoder_model = AutoModel.from_pretrained(model_name_or_path, return_dict=True, num_labels=num_classes)
    if use_multi_model:
        model = BiEncoderForClassificationMultiLayerMultiDim.from_pretrained(
            model_name_or_path, num_labels=num_classes, multi_layers=multi_layers, multi_dims=multi_dims
        )
    else:
        model = BiEncoderForClassification.from_pretrained(
            model_name_or_path, num_labels=num_classes
        )
    # No PEFT task type for auto model loading
    if use_peft:
        peft_config = LoraConfig(
            task_type=None, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1
        )
        model.encoder = get_peft_model(model.encoder, peft_config)
        model.encoder.print_trainable_parameters()

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    if DRY_RUN:
        train_dataset = eval_dataset

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=tokenized_collate_fn,
        train_dataset=train_dataset,
        # train_dataset=dataset["test"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=model.compute_metrics,
    )

    train_output = trainer.train()
    trainer.save_model()
    trainer.save_state()
    trainer.log_metrics("train_result", train_output.metrics)
    trainer.save_metrics("train_result", train_output.metrics)


    for split in dataset:
        print(split)
        if DRY_RUN and split == "train": continue
        # if split in {"train", "test"}: continue
        metrics = trainer.evaluate(dataset[split])
        trainer.save_metrics(split, metrics)

    # trainer.push_to_hub()

if __name__ == "__main__":
    # main()
    main_multi()

