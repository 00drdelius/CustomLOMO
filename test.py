from dataclasses import dataclass, field
from transformers import HfArgumentParser
from src.arguments import ModelArguments, DataArguments, MyTrainingArguments, WandbArguments
import sys
import os
from peft.tuners import lora
from peft import PeftModel
from torch.nn import modules
from transformers.models.llama import modeling_llama

parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments, WandbArguments))
print(parser)
model_args, data_args, train_args, wandb_args = parser.parse_yaml_file(yaml_file='config/args_lomo.yaml')

print(model_args, data_args, train_args, wandb_args)

