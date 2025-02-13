"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 messages_key='messages',  # Key for the messages list in the parquet file
                 max_length=1024,
                 truncation='error',
                 is_hf_dataset=False,
                 key="train"):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation
        # for simple support hf dataset

        if is_hf_dataset:
            from datasets import load_dataset 
            if isinstance(parquet_files, str) == True:
                parquet_files = [parquet_files] 
            hf_datasets = [load_dataset(parquet_file, split=key) for parquet_file in parquet_files] 
            self.messages = [] 
            for hf_dataset in hf_datasets:
                
                self.messages.extend(list(hf_dataset[messages_key])) # read on memory

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.messages_key = messages_key
        self.max_length = max_length

        if not is_hf_dataset:
            self._download()
            self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import pandas, numpy
            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        
        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]

        # First, get the full conversation tokens
        full_tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt', add_generation_prompt=False)
        input_ids = full_tokens[0]  # The output is already a tensor
        attention_mask = torch.ones_like(input_ids)
        
        # Create loss mask by identifying assistant responses
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
        
        # Process each message to find assistant responses
        current_length = 0
        for i, msg in enumerate(messages):
            # Get tokens for messages up to this point to find the start position
            prefix_messages = messages[:i+1]
            prefix_tokens = tokenizer.apply_chat_template(prefix_messages, tokenize=True, return_tensors='pt', add_generation_prompt=False)
            
            # Get tokens for messages up to previous point
            prev_tokens = tokenizer.apply_chat_template(messages[:i], tokenize=True, return_tensors='pt', add_generation_prompt=False) if i > 0 else None
            
            # Calculate start and end positions
            start_pos = prev_tokens[0].shape[0] if prev_tokens is not None else 0
            end_pos = prefix_tokens[0].shape[0]
            
            # If this is an assistant message, set loss mask
            if msg['role'] == 'assistant':
                loss_mask[start_pos:end_pos] = 1

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                        dtype=input_ids.dtype) * pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), 
                                              dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), 
                                         dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            elif self.truncation == 'error':
                raise ValueError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise ValueError(f'Unknown truncation method {self.truncation}')

        # Create position IDs
        # position_ids = torch.arange(len(input_ids), dtype=torch.long)
        position_ids = compute_position_id_with_mask(attention_mask)

        # Zero out position IDs for padding
        # position_ids = position_ids * attention_mask

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }