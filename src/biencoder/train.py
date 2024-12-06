"""Run BiEncoder Experiments

Run as:
accelerate launch src/biencoder/train.py \
--model_name_or_path "model_path" \
--output_model_name "dummy_model" \
--dataset_path "data_path" \
--batch_size 256
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Optional

import evaluate
import pandas as pd
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    HfArgumentParser,
)

from biencoder import (
    BiEncoder,
    BiEncoderForClassification,
    BiEncoderForClassificationMultiLayerMultiDim,
)
from biencoder_utils import print_eval_metrics

experiment_name = "<EXP_NAME>"

os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
print(f"Using {os.environ['MLFLOW_EXPERIMENT_NAME']=}")


@dataclass
class TrainArgs:
    model_name_or_path: str = field(metadata={"help": "location of the model"})
    output_model_name: str = field(
        metadata={"help": "output model name used to generate output path"}
    )
    dataset_path: str = field(metadata={"help": "location of the data"})
    file_suffix: str = field(
        default="", metadata={"help": "file_suffix for output folder"}
    )
    batch_size: int = field(default=64, metadata={"help": "batch size"})
    max_length: int = field(default=512, metadata={"help": "max length"})
    lr: float = field(default=3e-4, metadata={"help": "learning rate"})

    num_train_epochs: int = field(default=5, metadata={"help": "num epochs"})
    resume_from_checkpoint: bool = field(
        default=False, metadata={"help": "resume from checkpoint"}
    )

    use_peft: bool = field(default=False, metadata={"help": "use PEFT"})
    dry_run: bool = field(default=False, metadata={"help": "DRY RUN"})

    multi_layers: list[int] = field(
        default_factory=lambda: [2, 5, 8, 11],
        metadata={"help": "layers to use for biencoder"},
    )
    multi_dims: list[int] = field(
        default_factory=lambda: [64, 128, 384],
        metadata={"help": "dims to use for biencoder"},
    )

    freeze_encoder: bool = field(default=False, metadata={"help": "Freeze Encoder"})
    kldiv_loss: bool = field(default=False, metadata={"help": "KLDiv Loss"})
    bilinear_bias: bool = field(default=False, metadata={"help": "bilinear bias"})
    shared_bilinear: bool = field(
        default=False, metadata={"help": "share bilinear layer"}
    )
    matryoshka_bilinear: bool = field(
        default=False, metadata={"help": "use matryoshka bilinear"}
    )
    use_cosent_loss: bool = field(default=False, metadata={"help": "Use cosent loss"})
    use_multi_model: bool = field(
        default=True, metadata={"help": "Use Multi Dim Multi Layer Model"}
    )
    pooling_function_type: str = field(
        default="weighted_mean",
        metadata={"help": "Pooling function: cls_token or weighted_mean"},
    )

    normalize_temperature: Optional[float] = field(
        default=None, metadata={"help": "normalize temperature"}
    )
    use_prefix: Optional[str] = field(
        default='{"query": "query: ", "document": "passage: "}',
        metadata={"help": "query document prefixes"},
    )
    artifacts_dir: str = field(default="./", metadata={"help": "location of the model"})

    def __post_init__(self):
        if self.use_prefix:
            self.use_prefix = json.loads(self.use_prefix)
            assert (
                isinstance(self.use_prefix, dict)
                and "query" in self.use_prefix
                and "document" in self.use_prefix
            ), f"{self.use_prefix=} incorrect format. Should be dict with keys: query, document and prefix values for each key."
        else:
            self.use_prefix = None


parser = HfArgumentParser([TrainArgs])
train_args, unknown_args = parser.parse_args_into_dataclasses(
    return_remaining_strings=True
)
print(f"{train_args=}, {unknown_args=}")


model_name_or_path = train_args.model_name_or_path
output_model_name = train_args.output_model_name
dataset_path = train_args.dataset_path

batch_size = train_args.batch_size
max_length = train_args.max_length

num_train_epochs = train_args.num_train_epochs
resume_from_checkpoint = train_args.resume_from_checkpoint

DRY_RUN = train_args.dry_run
use_peft = train_args.use_peft
ARTIFACTS_DIR = train_args.artifacts_dir

lr = train_args.lr

multi_layers = train_args.multi_layers
multi_dims = train_args.multi_dims

freeze_encoder = train_args.freeze_encoder
kldiv_loss = train_args.kldiv_loss
bilinear_bias = train_args.bilinear_bias
shared_bilinear = train_args.shared_bilinear
matryoshka_bilinear = train_args.matryoshka_bilinear
use_cosent_loss = train_args.use_cosent_loss
pooling_function_type = train_args.pooling_function_type
normalize_temperature = train_args.normalize_temperature
use_multi_model = train_args.use_multi_model


use_prefix = train_args.use_prefix


if freeze_encoder:
    print(f"{freeze_encoder=}, {batch_size=}, {num_train_epochs=}")
    batch_size = batch_size * 8
    num_train_epochs = 1.0
    print(f"Updated {batch_size=}, {num_train_epochs=}")

if "marqo-gcl-e5-large-v2-130" in model_name_or_path:
    batch_size = 32

if "nomic" in model_name_or_path:
    use_prefix = dict(
        query="search_query: ",
        document="search_document: ",
    )

print(f"{use_multi_model=}, {use_prefix=}")

if not use_multi_model:
    output_model_name = f"{output_model_name}-single"
elif use_multi_model:
    output_model_name = f"{output_model_name}-multi"
    if kldiv_loss:
        output_model_name += "-kld"
    if not bilinear_bias:
        output_model_name += "-no_bilinear_bias"
    if shared_bilinear:
        output_model_name += "-shared_bilinear"
    if matryoshka_bilinear:
        output_model_name += "-matryoshka_bilinear"
if use_cosent_loss:
    output_model_name += "-cosent"
if pooling_function_type == "cls_token":
    output_model_name += "-cls_token"
if normalize_temperature is not None:
    output_model_name += f"-norm-temp-{normalize_temperature:.2f}"

if use_prefix:
    output_model_name += "-use_prefix"

if use_peft:
    output_model_name += "-peft"

if DRY_RUN:
    output_model_name += "_dry_run"
    num_train_epochs = 1

# output_dir = f"./models/{output_model_name}"
output_dir = os.path.join(ARTIFACTS_DIR, f"./models/{output_model_name}")
print(f"{output_dir=}")


print(f"{output_dir=}")


def transform_example(e):
    if use_prefix:
        return (
            f'{use_prefix["query"]}{e["query"]}',
            f'{use_prefix["document"]}{e["document"]}',
            e["labels"],
        )
    return (e["query"], e["document"], e["labels"])


def main_multi():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        do_train=True,
        do_eval=True,
        # eval_strategy="epoch",
        eval_on_start=True,
        # evaluation_strategy="steps",
        # save_strategy="epoch",
        # Save per steps
        eval_strategy="steps",
        eval_steps=20_000,
        save_strategy="steps",
        save_steps=20_000,
        save_total_limit=2,
        # dataloader_num_workers=10,
        remove_unused_columns=False,
        num_train_epochs=3.0 if DRY_RUN else num_train_epochs,
        max_steps=2 if DRY_RUN else -1,
        # max_steps=2, # Uncomment
        push_to_hub=False,
        hub_model_id=output_dir,
        load_best_model_at_end=True,
        report_to="mlflow",
        # label_smoothing_factor=0.1,
        label_names=["labels"],
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    with training_args.main_process_first():
        dataset = load_from_disk(dataset_path)
        num_classes = dataset["train"].features["labels"].num_classes
        print(dataset)
        print(f"{num_classes=}")

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

    # Make Model
    if use_multi_model:
        model = BiEncoderForClassificationMultiLayerMultiDim.from_pretrained(
            model_name_or_path,
            num_labels=num_classes,
            multi_layers=multi_layers,
            multi_dims=multi_dims,
            kldiv_loss=kldiv_loss,
            bilinear_bias=bilinear_bias,
            shared_bilinear=shared_bilinear,
            matryoshka_bilinear=matryoshka_bilinear,
            use_cosent_loss=use_cosent_loss,
            pooling_function_type=pooling_function_type,
            normalize_temperature=normalize_temperature,
            use_prefix=use_prefix,
            trust_remote_code=True,
        )
    else:
        model = BiEncoderForClassification.from_pretrained(
            model_name_or_path, num_labels=num_classes
        )

    if freeze_encoder:
        print(f"Freezing Encoder: {freeze_encoder=}")
        for param in model.encoder.parameters():
            param.requires_grad = False

    # No PEFT task type for auto model loading
    if use_peft:
        peft_config = LoraConfig(
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            use_rslora=True,
            target_modules=["_embeddings", "query", "value", "key", "dense"],
            modules_to_save=[
                "bilinear",
            ],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    if DRY_RUN:
        train_dataset = eval_dataset

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=tokenized_collate_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=model.compute_metrics,
    )

    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()
    trainer.log_metrics("train_result", train_output.metrics)
    trainer.save_metrics("train_result", train_output.metrics)

    if use_peft:
        with training_args.main_process_first():
            model = model.merge_and_unload()
            merged_path = f"{output_dir}/merged"
            model.save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)

    for split in dataset:
        print(split)
        # if split == "train": continue
        if DRY_RUN and split == "train":
            continue
        predictions, label_ids, metrics = trainer.predict(
            dataset[split], metric_key_prefix=split
        )
        trainer.save_metrics(split, metrics)
        if trainer.is_world_process_zero():
            print(f"{label_ids.shape=}")
            all_predictions = model.parse_predictions((predictions, label_ids))
            df_t = pd.DataFrame(all_predictions)
            df_t["label_ids"] = label_ids
            df_t.to_csv(Path(output_dir) / f"{split}_predictions.csv", index=False)

    with training_args.main_process_first():
        print_eval_metrics(output_dir)


if __name__ == "__main__":
    main_multi()
