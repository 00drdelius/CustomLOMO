import os
import copy
import re
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm


IGNORE_INDEX = -100
REPRODUCIBILITY_SEED = 0


class CustomDataset(Dataset):
    def __init__(self, data_args, tokenizer, split) -> None:
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.split = split
        self.sample_size = 1000
        self.max_length = data_args.data_max_length
        
        save_dir = os.path.join(data_args.data_dir, data_args.data_filename.split(".")[0])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        save_file = os.path.join(save_dir, f"{split}.pt")
        if data_args.refresh or not os.path.exists(save_file):
            dataset = load_dataset('json', data_files=f"{data_args.data_dir}/{data_args.data_filename}", split=split, streaming=True)
            self.datas = self.process(dataset, save_file)
        else:
            print('Loading data from', save_file)
            self.datas = torch.load(save_file)

        print("Data size:", len(self.data))
        print("Data format:", self.data[0])
        print("Max length:", max([len(d['input_ids']) for d in self.data])) if self.split == 'train' else \
            print('Max length:', max([max([len(d) for d in dd['input_ids']]) for dd in self.data]))

    def process(self, dataset, save_file):
        datas = []
        for data in dataset:
            instruction = data['Instruction']
            inputing = data['Input']
            response = data['Response']

            def _tokenize_fn(instruction, inputing, response):
                example = "USER:%s\n%s\nASSISTANT:%s" % (instruction, inputing, response)
                example_tokenized = self.tokenizer.encode(example, truncation=True, max_length = self.data_args.data_max_length)
                example_tokenized += [self.tokenizer.eos_token_id]
                instruction_tokenized = self.tokenizer.encode(re.search(r"(.|\n)*ASSISTANT:", example).group())

                input_ids = example_tokenized
                labels = copy.deepcopy(input_ids)
                if not self.data_args.train_on_inputs:
                    labels = np.array(labels)
                    labels[:len(instruction_tokenized) -1] = IGNORE_INDEX
                return input_ids, labels
        
            input_ids, labels = _tokenize_fn(instruction, inputing, response)

            datas.append({
                "input_ids": input_ids,
                "labels": labels,
                "Instruction": instruction,
                "Input": inputing,
                "Response": response
            })

        if self.sample_size > 0 and len(data) > self.sample_size:
            random.seed(REPRODUCIBILITY_SEED)
            possible_idxs = list(range(len(data)))
            sampled_idxs = random.sample(possible_idxs, self.sample_size)
            data = [data[i] for i in sampled_idxs]
            print(f'Sampled {self.sample_size} examples from {len(possible_idxs)} examples.')

        torch.save(datas, save_file)
        print("Saving data to", save_file)
        return datas
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Any:
       return {
           'input_ids':self.datas[index]['input_ids'],
           'labels':self.datas[index]['labels']
       }
    
if __name__ == '__main__':
    from transformers import HfArgumentParser
    from arguments import ModelArguments, DataArguments
    from transformers import AutoTokenizer

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    model_args.model_name_or_path = './llamaConfig'
    data_args.refresh = True
    train_on_inputs = False
    data_args.data_max_length = 512
    data_args.data_dir = 'dataset'
    data_args.data_filename = 'test.jsonl'

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        padding_side='left',
    )
    tokenizer.pad_token_id = 0

    train_dataset = CustomDataset(data_args, tokenizer, split='train')
    # eval_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.eval_split)
    # test_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.test_split)
    print(train_dataset.data[0])

