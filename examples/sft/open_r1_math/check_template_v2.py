import logging

from verl.utils.dataset.chat.chat_template import  (
    IGNORE_TOKEN_ID,
    ChatTemplatePrompter,
    ChatTemplateStrategy,
    load)
from verl.utils.dataset.chat.chat_template_utils import get_chat_template
from transformers import AutoTokenizer
from addict import Dict

if __name__ == "__main__":
    llama3_tokenizer = AutoTokenizer.from_pretrained("/home/pvduy/tue/ii_verl_repo/ii_verl/examples/sft/open_r1_math/qwen_r1_7b")
    
    from datasets import Dataset, load_dataset
    from datasets import enable_caching
    enable_caching()

    ds = load_dataset("tuenguyen/open-r1-math-220k-chatml", split="train")
    ds = ds.select(range(10))
    import os
    def tokenizer_prompt(sample):
        messages = sample['messages']
        inputs_sample = llama3_tokenizer.apply_chat_template(messages)
        # print(inputs_sample)
        return {
            "text": inputs_sample
        }
    ds = ds.map(tokenizer_prompt, num_proc=os.cpu_count())
    print(ds)
    print(llama3_tokenizer.decode(ds[0]['text'], skip_special_tokens=False))