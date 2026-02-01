from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from redis import Redis


system_prompt = """
You are a helpful assistant that can answer questions about mathematics. For every question, output your thinking process in <think></think> and the final answer in <answer></answer>. If you are uncertain about or wish to spend more compute on a particular direction, output "HIGH" and then continue your reasoning.
"""

system_prompt_message = {"role": "system", "content": system_prompt}

class MathsDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self._redis = None
        self._idx = 0
    
    @property
    def redis(self):
        if self._redis is None:
            self._redis = Redis(host="localhost", port=6379, db=0)
        return self._redis
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        problem = item["problem"]
        answer = item["answer"]
        self.redis.set(f"correct_answer:{idx}", answer)
        return idx, problem, answer
    
    def __next__(self):
        if self._idx >= len(self):
            self._idx = 0
        idx, problem, answer = self[self._idx]
        self._idx += 1
        return idx, problem, answer
    
    def __iter__(self):
        return self