from verl import DataProto
from rl_verifier import RLVerifierClient
import torch


class APIRewardManager():
    """Reward model as API.
    """

    def __init__(self, tokenizer, api_url, max_workers = 10, timeout = 30, verification_info_column = 'verification_info'):
        self.tokenizer = tokenizer
        self.client = RLVerifierClient(api_url, timeout=timeout)
        self.max_workers = max_workers
        self.verification_info_column = verification_info_column
        
    def __call__(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        lst_request_items = []
        lst_score_positions = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            llm_output = self.tokenizer.decode(sequences)

            verification_info = data_item.non_tensor_batch.get(self.verification_info_column)
            if not verification_info:
                raise ValueError(f"Verification info column {self.verification_info_column} not found in data")
            
            lst_request_items.append(
                (llm_output, verification_info)
            )
            lst_score_positions.append(valid_response_length - 1)

        # Call API to get reward scores
        scores = self.client.verify_batch(
            batch=lst_request_items,
            max_workers=self.max_workers,
            default_value=0.0,
            progress_bar=True
        )
        
        print("**Accuracy**:", sum(scores) / len(scores))
        
        for i, score in enumerate(scores):
            reward_tensor[i, lst_score_positions[i]] = score

        return reward_tensor