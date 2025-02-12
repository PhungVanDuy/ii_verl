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
    llama3_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    # llama3_tokenizer.add_special_tokens({"eos_token": "<|eot_id|>"})
    chat_template = llama3_tokenizer.chat_template
    strategy = load(
        llama3_tokenizer,
        Dict(
            {
                "train_on_inputs": False,
                "train_on_eos": "all",
                "sequence_len": 32000,
            }
        ),
        Dict(
            {
                "chat_template": "tokenizer_default",
                "message_field_role": "role",
                "message_field_content": "content",
                "roles": {
                    "user": ["user"],
                    "assistant": ["assistant"],
                    "system": ["system"],
                },
                "field_messages": "messages",
            }
        ),
    )
    from datasets import Dataset, load_dataset
    from datasets import enable_caching
    enable_caching()

    ds = load_dataset("tuenguyen/open-r1-math-220k-chatml-v2", split="train")
    ds = ds.select(range(10))
    # print(ds[0])
    ix = 0
    print(ds[ix])
    res = strategy.tokenize_prompt(ds[ix])
    # print(res)
    input_ids = [i for i, v in zip(res["input_ids"], res['labels']) if v != IGNORE_TOKEN_ID]
    print(llama3_tokenizer.decode(input_ids, skip_special_tokens=False))
    ds = ds.map(strategy.tokenize_prompt)
    print(ds)
    print(llama3_tokenizer.decode(ds[ix]['input_ids'], skip_special_tokens=False))
    ds = load_dataset("tuenguyen/open-r1-math-220k-chatml", split="train")
    ds = ds.map(strategy.tokenize_prompt, num_proc=32)
    ds = ds.map(lambda x: {"length": len(x['input_ids'])}, num_proc=32)
    a = list(set(ds['length']))
    print(max(a), min(a), sum(a)/len(a))

    
    # print(ds['length'].min(), ds['length'].max(), ds['length'].mean())

    
