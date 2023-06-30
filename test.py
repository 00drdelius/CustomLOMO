from src.mydatasets import MyDataset
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import sys
import os


sys.path.append(os.getcwd())


tokenizer_path = "baichuanConfig"


@dataclass
class DatasetArgs:
    data_dir:str = field(metadata={'help':'your data directory, not to file but to path'})
    dataset_name:str = field(metadata={'help':"dataset name"})
    # data_tag:str = field(metadata={'help':'data tag'})
    split:str = field(default='train', metadata={'help':"the split name of dataset"})
    data_max_length:int = field(metadata={'help':"max length of input sequence"})
    refresh:bool = field(default=False)

parser = HfArgumentParser((DatasetArgs))
# info = parser.parse_args_into_dataclasses()


# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code =True)
# tokenizer.pad_token_id = 0

# dataset = MyDataset(info, tokenizer)
dataset = load_dataset('json', data_files='dataset/test.jsonl', keep_in_memory=True)

for i in dataset['train']:
    print(i)

