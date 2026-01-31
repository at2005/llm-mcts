from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from redis import Redis

system_prompt = """
You are a helpful assistant that can answer questions about mathematics. For every question, output your thinking process in <think></think> and the final answer in <answer></answer>.
"""

system_prompt_message = {"role": "system", "content": system_prompt}

class MathsDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.redis = Redis(host="localhost", port=6379, db=0)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):

        item = self.ds[idx]
        problem = item["problem"]
        answer = item["answer"]
        self.redis.set(f"correct_answer:{idx}", answer)
        return idx, problem, answer

def maths_dataloader():
    print("Loading dataset...")
    ds = MathsDataset(load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train"))
    print("Dataset loaded")
    dataloader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4)
    return dataloader