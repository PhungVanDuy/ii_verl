from typing import List, Union, Dict

import pandas as pd

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer

from verl.utils.dataset.chat.chat_template import  (
    IGNORE_TOKEN_ID,
    ChatTemplatePrompter,
    ChatTemplateStrategy,
    load)

class SFTChatDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        split: str,
        tokenizer: PreTrainedTokenizer,

        # config for dataset column names   

        field_messages: str = 'messages',
        message_field_role: str = 'role',
        message_field_content: str = 'content',

        # config for training on turn
        train_on_inputs: bool = False,
        train_on_eos: str = "turn", # all, turn, last, none
        roles:  Dict[str, List[str]]=None, # {"user": ["user"], "assistant": ["assistant"]}

        # config for sequence length
        sequence_len: int = 512,

        chat_template:str="qwen25", # supported chat templates: alpaca, mistral_v1, mistral_v2v3, mistral_v3_tekken, chatml, gemma, cohere, llama3, llama3_2_vision, phi_3, phi_35, deepseek_v2, jamba, qwen_25, exaone, metharme
    ):
        
        super().__init__()

        self.dataset = load_dataset(path)
        if split not in self.dataset:
            raise ValueError(f"Split {split} not found in dataset {path}")
        self.dataset = self.dataset[split]

        # we will build the prompt_strategy
        from addict import Dict as DefaultDict
        strategy = load(
            tokenizer,
            DefaultDict(
                {
                    "train_on_inputs": train_on_inputs,
                    "train_on_eos": train_on_eos,
                    "sequence_len": sequence_len,
                }
            ),
            DefaultDict(
                {
                    "chat_template": chat_template,
                    "message_field_role": message_field_role,
                    "message_field_content": message_field_content,
                    "roles": roles,
                    "field_messages": field_messages,
                }
            )
        )
            # res = strategy.tokenize_prompt(ds[0])

        # after that we tokenizer dataset

        self.dataset = self.dataset.map(strategy.tokenize_prompt)
        sample_0 = self.dataset[0]
        input_ids = sample_0['input_ids']
        print(tokenizer.decode(input_ids))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row =  self.dataset[idx]
        input_ids = torch.tensor(row['input_ids']).long()
        attention_mask = torch.tensor(row['attention_mask']).long()
        labels = torch.tensor(row['labels']).long()
        
        position_ids = compute_position_id_with_mask(attention_mask)
        loss_mask = torch.where(
            labels != IGNORE_TOKEN_ID,
            torch.ones_like(labels), # 1 for loss and 0 for mask
            torch.zeros_like(labels)
        )

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }

