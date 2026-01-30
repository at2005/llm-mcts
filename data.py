from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class MathsDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        problem = item["problem"]
        answer = item["answer"]
        return idx, problem, answer

def maths_dataloader():
    ds = MathsDataset(load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train"))
    dataloader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4)
    return dataloader