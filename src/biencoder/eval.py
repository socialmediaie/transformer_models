"""Evaluate BiEncoder Models

accelerate launch src/biencoder/eval.py \
--model_name_or_path "model_name" \
--dataset_paths "data_path" \
--dataset_names original

"""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from biencoder import (
    BiEncoderForClassificationMultiLayerMultiDim,
)
from biencoder_utils import print_eval_metrics
from datasets import load_from_disk
from transformers import AutoTokenizer, Trainer, TrainingArguments, HfArgumentParser


@dataclass
class EvalArgs:
    model_name_or_path: str = field(metadata={"help": "location of the model"})
    file_suffix: str = field(
        default="", metadata={"help": "file_suffix for output folder"}
    )
    batch_size: int = field(default=64, metadata={"help": "batch size"})
    max_length: int = field(default=512, metadata={"help": "max length"})
    dataset_paths: list[str] = field(
        default_factory=list, metadata={"help": "datasets used for the model"}
    )
    dataset_names: list[str] = field(
        default_factory=list, metadata={"help": "datasets names for each dataset"}
    )

    def __post_init__(self):
        if not self.dataset_names:
            self.dataset_names = [
                f"dataset_{i}" for i in range(len(self.dataset_paths))
            ]
        assert len(self.dataset_paths) == len(
            self.dataset_names
        ), f"{len(self.dataset_paths)=} != {len(self.dataset_names)=}"
        self.datasets = dict(zip(self.dataset_names, self.dataset_paths))


parser = HfArgumentParser([EvalArgs])
eval_args, unknown_args = parser.parse_args_into_dataclasses(
    return_remaining_strings=True
)
print(f"{eval_args=}, {unknown_args=}")


model_name_or_path = eval_args.model_name_or_path
file_suffix = eval_args.file_suffix
batch_size = eval_args.batch_size
max_length = eval_args.max_length
eval_datasets = eval_args.datasets


output_dir = model_name_or_path.replace("/models/", "/evals/")
output_dir += file_suffix
print(f"{output_dir=}")


def main():
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False,
        do_eval=True,
        dataloader_num_workers=10,
        report_to="none",
        # local_rank=0
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = BiEncoderForClassificationMultiLayerMultiDim.from_pretrained(
        model_name_or_path,
    )

    transform_example = (
        BiEncoderForClassificationMultiLayerMultiDim.get_transform_example_fn(
            use_prefix=model.use_prefix
        )
    )

    def tokenized_collate_fn(examples):
        queries, documents, labels = zip(*(transform_example(e) for e in examples))
        query_encoding = tokenizer(
            list(queries),
            truncation=True,
            max_length=max_length,
            padding="longest",
            return_tensors="pt",
        )
        document_encoding = tokenizer(
            list(documents),
            truncation=True,
            max_length=max_length,
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

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=model.compute_metrics,
        data_collator=tokenized_collate_fn,
        tokenizer=tokenizer,
    )

    for dataset_key, dataset_path in eval_datasets.items():
        dataset = load_from_disk(dataset_path)
        print(dataset)
        for split in dataset:
            metric_key_prefix = f"{dataset_key}__{split}"
            print(f"{metric_key_prefix=}")
            if dataset[split].num_rows > 10_000_000:
                print(
                    f"Skipping {metric_key_prefix=} as {dataset[split].num_rows=} > 10M."
                )
                continue
            predictions, label_ids, metrics = trainer.predict(
                dataset[split], metric_key_prefix=metric_key_prefix
            )
            trainer.save_metrics(metric_key_prefix, metrics)
            if trainer.is_world_process_zero():
                print(f"{label_ids.shape=}")
                all_predictions = model.parse_predictions((predictions, label_ids))
                df_t = pd.DataFrame(all_predictions)
                df_t["label_ids"] = label_ids
                df_t.to_csv(
                    Path(output_dir) / f"{metric_key_prefix}_predictions.csv",
                    index=False,
                )
                print(f"{metrics=}")

    with training_args.main_process_first():
        print_eval_metrics(output_dir, file_suffix=file_suffix)


if __name__ == "__main__":
    main()
