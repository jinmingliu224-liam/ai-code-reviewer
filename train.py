import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="bigcode/starcoder")
    use_auth_token: bool = field(default=False)
    trust_remote_code: bool = field(default=False)


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    preprocessing_num_workers: Optional[int] = field(default=None)
    block_size: Optional[int] = field(default=2048)
    percentage: Optional[int] = field(default=None)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_auth_token=model_args.use_auth_token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        use_auth_token=model_args.use_auth_token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        raw_datasets = load_dataset("json", data_files=data_files)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=data_args.block_size)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir):
        checkpoint = get_last_checkpoint(training_args.output_dir)

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
