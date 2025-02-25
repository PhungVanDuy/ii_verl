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
    llama3_tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")
    # llama3_tokenizer.add_special_tokens({"eos_token": "<|eot_id|>"})
    chat_template = llama3_tokenizer.chat_template
    strategy = load(
        llama3_tokenizer,
        Dict(
            {
                "train_on_inputs": False,
                "train_on_eos": "all",
                "sequence_len": 512,
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
    print(strategy)
    sample = [{
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "What is today's stock price of Apple?",
            },
            {
                "role": "assistant",
                "content": "The stock price of Apple is $123.45.\n",
            },
        ]
    }]
    from datasets import Dataset
    ds = Dataset.from_list(sample)
    print(ds[0])
    res = strategy.tokenize_prompt(ds[0])
    print(res)
    input_ids = [i for i, v in zip(res["input_ids"], res['labels']) if v != IGNORE_TOKEN_ID]
    print(llama3_tokenizer.decode(input_ids, skip_special_tokens=False))
    ds = ds.map(strategy.tokenize_prompt)
    print(ds)
    print(llama3_tokenizer.decode(ds[0]['input_ids'], skip_special_tokens=False))

    