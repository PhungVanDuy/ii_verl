# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import logging
import re
from contextlib import nullcontext
import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from tensordict import TensorDict
from torch.utils.data import DataLoader, DistributedSampler
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.dataset import SFTDataset
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.tracking import Tracking
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, set_ulysses_sequence_parallel_group
from torch.distributed.device_mesh import DeviceMesh

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage
from peft import LoraConfig, TaskType, get_peft_model

from verl.workers.sharding_manager import FSDPUlyssesShardingManager
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl import DataProto
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer, BaseSFTTrainer
from verl.utils.dataset.sft_chat_dataset import SFTChatDataset
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))

# Here we use collator padding for speed training

def padding_collator(batch, tokenizer):
    keys = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "loss_mask"
    ]
    
    max_length = max([len(item['input_ids']) for item in batch]) 
    return_dict_batch = {
        "input_ids": torch.zeros((len(batch), max_length), dtype=torch.long) + tokenizer.pad_token_id,
        "attention_mask": torch.zeros((len(batch), max_length), dtype=torch.long),
        "position_ids": torch.zeros((len(batch), max_length), dtype=torch.long),
        "loss_mask": torch.zeros((len(batch), max_length), dtype=torch.long)
    }
    for i, item in enumerate(batch):
        for key in keys:
            return_dict_batch[key][i, :len(item[key])] = item[key]
    return return_dict_batch

class FSDPSFTTrainerHF(BaseSFTTrainer):
    def __init__(self, config, device_mesh, ulysses_device_mesh):
        super().__init__(config, device_mesh, ulysses_device_mesh)
    
    # override the _build_dataloader method
    # @override
    def _build_dataloader(self):

        config = self.config
        # build dataset
        # # first we load train dataset 
        # path: str,
        # split: str,
        # tokenizer: PreTrainedTokenizer,

        # # config for dataset column names   

        # field_messages: str = 'messages',
        # message_field_role: str = 'role',
        # message_field_content: str = 'content',

        # # config for training on turn
        # train_on_inputs: bool = False,
        # train_on_eos: str = "turn", # all, turn, last, none
        # roles:  Dict[str, List[str]]=None, # {"user": ["user"], "assistant": ["assistant"]}

        # # config for sequence length
        # sequence_len: int = 512,

        # chat_template:str="qwen25",
        self.train_dataset = SFTChatDataset(
            path=config.data.train_path,
            split=config.data.train_split,
            tokenizer=self.tokenizer,
            field_messages=config.data.field_messages,
            message_field_role=config.data.message_field_role,
            message_field_content=config.data.message_field_content,
            train_on_inputs=config.data.train_on_inputs,
            train_on_eos=config.data.train_on_eos,
            roles=config.data.roles,
            sequence_len=config.data.sequence_len,
            chat_template=config.data.chat_template)
        

        self.val_dataset = SFTChatDataset(
            path=config.data.val_path,
            split=config.data.val_split,
            tokenizer=self.tokenizer,
            field_messages=config.data.field_messages,
            message_field_role=config.data.message_field_role,
            message_field_content=config.data.message_field_content,
            train_on_inputs=config.data.train_on_inputs,
            train_on_eos=config.data.train_on_eos,
            roles=config.data.roles,
            sequence_len=config.data.sequence_len,
            chat_template=config.data.chat_template)
        
        # build dataloader
        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank('dp')
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f'Using SP rank {rank} and size {world_size} for data distribution')
                print(f'Each SP rank gets different data, but the same data WITHIN the same rank')
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f'Using FSDP rank {rank} and size {world_size} for data distribution')
        from functools import partial
        self.train_sampler = DistributedSampler(self.train_dataset,
                                                shuffle=True,
                                                num_replicas=world_size,
                                                rank=rank,
                                                drop_last=True)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=config.data.train_batch_size,
                                           sampler=self.train_sampler,
                                           num_workers=8,
                                           pin_memory=True,
                                           drop_last=True,
                                           collate_fn=partial(padding_collator, tokenizer=self.tokenizer))

        self.val_sampler = DistributedSampler(self.val_dataset,
                                              shuffle=True,
                                              num_replicas=world_size,
                                              rank=rank,
                                              drop_last=True)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=config.data.micro_batch_size_per_gpu,
                                         sampler=self.val_sampler,
                                         num_workers=8,
                                         pin_memory=True,
                                         drop_last=True,
                                         collate_fn=partial(padding_collator, tokenizer=self.tokenizer))


from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
import hydra

from torch.distributed.device_mesh import init_device_mesh

from verl.utils.distributed import initialize_global_process_group

@hydra.main(config_path='config', config_name='sft_trainer_hf', version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('fsdp',))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type='cuda',
                                           mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
                                           mesh_dim_names=('dp', 'sp'))
    trainer = FSDPSFTTrainerHF(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh)
    trainer.fit()


if __name__ == '__main__':
    main()
