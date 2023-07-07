import copy
import os
import sys

from random import sample

import torch
from torch.utils.data import Subset
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from transformers import set_seed
from dataclasses import asdict
#from transformers.deepspeed import HfDeepSpeedConfig
from peft import get_peft_model, TaskType, LoraConfig, prepare_model_for_kbit_training
import wandb
# os.environ['WANDB_MODE'] = 'debug'

python_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print("PYTHON_PATH", python_path)
sys.path.append(python_path)
from log import print
from arguments import ModelArguments, DataArguments, MyTrainingArguments, WandbArguments
from mydatasets import MyDataset, get_dataset_info
from deliusdatasets import CustomDataset
from lomo_qlora_trainer import LoRATrainer
from utils import DataCollatorForCauselLM, EvalDataCollatorForCauselLM, find_all_linear_names
from QLoRACollator import SFTDataCollator
from loss import TargetLMLoss

def compute_metrics(all_pred, eval_dataset, eval_prefix=None):
    golds = [ins['answer'] for ins in eval_dataset.data]
    preds = all_pred[:len(golds)]

    acc = round(sum([int(pred == gold) for pred, gold in zip(preds, golds)]) / len(golds), 6)
    result = {'acc': acc}
    return result


def train():
    # ========== 1. logs and args ==========
    #torch.set_default_dtype(torch.bfloat16)
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments, WandbArguments))
    if sys.argv[-1].endswith(".yaml"):
        model_args, data_args, training_args, wandb_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args, wandb_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)


    # ========== 2. Load pretrained model and tokenizer. ==========
    #ds_config = training_args.deepspeed
    #dschf = HfDeepSpeedConfig(ds_config)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.gradient_checkpointing = training_args.gradient_checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,
        config=config,
        device_map="auto",
        load_in_4bit = True,
        torch_dtype = torch.float16,
        trust_remote_code = True,
        #-----------------------------------------Added--------------------------------------
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
    )
        
    #QLoRA quantization config
    # casts all the non int8 modules to full precision (fp32) for stability
    model = prepare_model_for_kbit_training(model)
    
    modules = find_all_linear_names(model)
    # use peft
    if training_args.peft_type is not None:
        print(f'Using peft.{training_args.peft_type}')
        if training_args.peft_type == 'lora':
            peft_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                # target_modules=["q_proj", "v_proj"],
                target_modules=modules,
                lora_dropout=training_args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            model.enable_input_require_grads()
        else:
            raise ValueError(f"Unknown PEFT type: {training_args.peft_type}")
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.config.torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
    )
    tokenizer.pad_token_id = 0
    loss_func = TargetLMLoss(ignore_index=tokenizer.pad_token_id)

    # ========== 3. Preprocessing the datasets. ==========
    train_dataset = CustomDataset(data_args, tokenizer, split='train')

    #eval_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.eval_split)
    # if dataset_info.test_split:
    #     test_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.test_split)
    #     eval_dataset = {
    #         # 'validation': eval_dataset,
    #         'test': test_dataset
    #     }

    # ========== 4. Initialize our Trainer. ==========
    trainer = LoRATrainer(
        model=model,
        args=training_args,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=data_args.data_max_length),
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_loss=loss_func
    )
    print("*** starting training ***")
    train_result = trainer.train()
    # 保存最好的checkpoint
    final_save_path = os.path.join(training_args.output_dir, 'final')
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    train()
