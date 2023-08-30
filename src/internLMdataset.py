from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from typing import Union
import regex as re
import torch
import copy
import numpy as np
import random
from dataclasses import field, dataclass

path = Path("InternLMCode").absolute()
dataPath = path.joinpath("dataset","test.jsonl")
tokenizerPath = path.joinpath("tokenizer")
# print(list(dataset)[0])

REPRODUCIBILITY_SEED = 0
IGNORE_INDEX = -100
INPUTKEY = "<|User|>"
OUTPUTKEY = "<|Bot|>"
USER = "{inputKey}{Instruction}{Input}"
BOT = "{outputKey}{Response}"

class CustomDataset(Dataset):

    INTERNLM_SPECIAL_MAP = {"<eoh>": 103167, "<eoa>": 103166}

    def __init__(self, tokenizer, data_args, data_filePath:Union[Path, str]) -> None:
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        if isinstance(data_filePath, Path):
            cacheFilePath = data_filePath.parent.joinpath(data_filePath.name.split(".")[0]+".pt")
        else:
            cacheFilePath = Path(data_filePath).parent.joinpath(Path(data_filePath).name.split('.')[0]+".pt")
        self.sample_size = 1000
        if cacheFilePath.exists() and not data_args.refresh:
            self.datas = torch.load(cacheFilePath)
        else:
            dataset = load_dataset("json", data_files=str(data_filePath), streaming=True, split='train')
            self.datas = self.process(dataset, cacheFilePath)
    
    def process(self, dataset, saveDir):
        datas = []
        for data in dataset:
            #data: type:dict, element: {"Instruction":..., "Input":..., "Response":...}
            input_ids, labels = self._tokenizer_fn(**data, special_tokens_map=self.INTERNLM_SPECIAL_MAP)
            datas.append(dict(
                input_ids=input_ids,
                labels=labels,
                **data
            ))

        if self.sample_size > 0 and len(datas) > self.sample_size:
            random.seed(REPRODUCIBILITY_SEED)
            possible_idxs = list(range(len(datas)))
            sampled_idxs = random.sample(possible_idxs, self.sample_size)
            datas = [datas[i] for i in sampled_idxs]
            print(f'Sampled {self.sample_size} examples from {len(possible_idxs)} examples.')

        torch.save(datas, saveDir)
        print("Saving data to", saveDir)
        return datas
            
    # before_tokenized: "<|User|>{prompt}<eoh>\n<|Bot|>{response}<eoa>"
    def _tokenizer_fn(self, Instruction, Input, Response, special_tokens_map=None):
        '''
        special_tokens_map: currently used for internLM's tokens: <eoh>;<eoa>
        '''
        sample = {"User":..., "Bot":...} # "User": "<|User|>(prompt)", "Bot": "<|Bot|>(response)"
        Input = Input.replace(" ", "")
        sample["User"] = USER.format(
            inputKey=INPUTKEY,Instruction = Instruction, Input="\n"+Input
        ) if Input else USER.format(inputKey=INPUTKEY,Instruction = Instruction, Input=Input)

        sample['Bot'] = BOT.format(outputKey=OUTPUTKEY, Response = Response)
        prompt_id = self.tokenizer.encode(sample['User'])
        response_id = self.tokenizer.encode(sample['Bot'])
        if special_tokens_map:
            prompt_id+=[special_tokens_map['<eoh>']]
            response_id+=[special_tokens_map['<eoa>']]

        input_ids = prompt_id+response_id+[self.tokenizer.eos_token_id]
        input_ids = input_ids[:self.data_args.data_max_length]
        matcher = f"(?<={re.escape(OUTPUTKEY)})(.|\r\n|\n)*"
        pure_response = re.compile(matcher).search(sample["Bot"]).group()
        tk_pure_res = self.tokenizer.encode(pure_response, truncation=True, max_length=self.data_args.data_max_length)

        labels = copy.deepcopy(input_ids)
        if not self.data_args.train_on_inputs:
            labels = np.array(labels)
            if special_tokens_map:
                labels[:len(input_ids)-len(tk_pure_res)-1] = IGNORE_INDEX # <eoa> the last
                labels[-1] = IGNORE_INDEX
            else:
                labels[:len(input_ids)-len(tk_pure_res)] = IGNORE_INDEX

        return input_ids, labels

    def __getitem__(self, index):
        if self.data_args.debug == True:
            return self.datas[index]
        else:
            return {
                'input_ids':self.datas[index]['input_ids'],
                'labels':self.datas[index]['labels']
            }

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    from rich import print
    @dataclass
    class TrainArguments:
        debug:bool=field()
        refresh:bool = field()
        train_on_inputs:bool=field(default=False)
        data_max_length:int=field(default=8192)
    
    data_args = TrainArguments(debug=True, refresh=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizerPath), trust_remote_code=True)
    dataset = CustomDataset(tokenizer, data_args, dataPath)
    test = dataset[-1]
    input_ids = test['input_ids']
    labels = list(filter(lambda x: x!=-100, test['labels']))
    regress = tokenizer.decode(input_ids)
    labels_regress = tokenizer.decode(labels)
    print("input ids: ", regress, "\n\n")
    print("labels: ", labels_regress, "\n\n")








    
    

