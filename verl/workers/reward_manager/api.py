"""
    Client function to compute reward scores for a batch of data by calling a remote API server.
    
    Expected API Server Design:
    - Endpoint: POST {api_url}. Example: http://localhost:8000/reward
    - Input JSON Format:
        {
            "solutions": [str, ...],      # List of solution to evaluate
            "ground_truths": [str, ...]   # List of corresponding ground truth
        }
    - Expected Response:
        {
            "scores": [float, ...]        # List of reward scores between 0.0 and 1.0
        }
"""


from verl import DataProto
from typing import List
import torch
import requests


def compute_batch_scores(lst_solution_strs: List[str], lst_ground_truths: List[str], api_url: str) -> List[float]:
    payload = {
        "solutions": lst_solution_strs,
        "ground_truths": lst_ground_truths,
    }
    
    response = requests.post(api_url, json=payload)
    response.raise_for_status()
    
    return response.json()['scores']
class RewardAPIManager():
    """Reward model as API.
    """

    def __init__(self, tokenizer, api_url) -> None:
        self.tokenizer = tokenizer
        self.api_url = api_url
        
    def __call__(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        lst_solution_strs = []
        lst_ground_truths = []
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
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch.get('ground_truth')
            lst_solution_strs.append(sequences_str)
            lst_ground_truths.append(ground_truth)
            lst_score_positions.append(valid_response_length - 1)
        
        # Call API to get reward scores
        scores = compute_batch_scores(lst_solution_strs, lst_ground_truths, self.api_url)
        for i, score in enumerate(scores):
            reward_tensor[i, lst_score_positions[i]] = score
        
        return reward_tensor