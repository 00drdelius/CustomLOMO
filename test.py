from dataclasses import dataclass, field
from transformers import HfArgumentParser
from transformers import Trainer
from src.arguments import ModelArguments, DataArguments, MyTrainingArguments, WandbArguments
import sys
import os



parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments, WandbArguments))
print(parser)
model_args, data_args, train_args, wandb_args = parser.parse_yaml_file(yaml_file='config/args_lomo.yaml')

print(model_args, data_args, train_args, wandb_args)

