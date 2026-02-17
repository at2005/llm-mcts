from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from redis import Redis
import re
from typing import Optional

system_prompt = """You are a mathematics reasoning assistant. Your job is to solve math problems correctly and clearly.
OUTPUT FORMAT (strict):
- Produce one or more reasoning steps. Each step MUST be wrapped exactly like:
<|start_step|>...<|end_step|>
- After the final step, output the final result exactly once, wrapped like:
<answer>...</answer>
Rules:
1) Every message you produce must follow the format above: steps first, then a single <answer> block.
2) Do NOT put anything outside step blocks except the final <answer>...</answer>.
3) Each step should be concise, logically ordered, and sufficient to justify the final answer.
4) The final answer inside <answer>...</answer> MUST be exactly one boxed LaTeX expression, formatted as <answer>\\boxed{...}</answer>. Do not include extra text, and do not use unmatched $ delimiters.
"""

system_prompt_message = {"role": "system", "content": system_prompt}


def parse_after_hashes(text: str) -> Optional[str]:
    """
    Returns the content after a line starting with '####'.
    Example: '#### 624' -> '624'
    Returns None if no such line exists.
    """
    m = re.search(r"(?m)^\s*####\s*(.*)$", text)
    return m.group(1).strip() if m else None


def get_dataset(dataset_name: str, seed: Optional[int] = None):
    if dataset_name == "openai/gsm8k":
        return GSM8KMathsDataset(load_dataset(dataset_name, "main", split="train"), seed=seed)
    elif dataset_name == "EleutherAI/hendrycks_math":
        return MATHDataset(load_dataset(dataset_name, "algebra", split="train"), seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

class MATHDataset(Dataset):
    def __init__(self, ds, seed: Optional[int] = None):
        if seed is not None:
            ds = ds.shuffle(seed=seed)
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
        answer = item["solution"]
        return idx, problem, answer
    
    def __next__(self):
        if self._idx >= len(self):
            self._idx = 0
        idx, problem, answer = self[self._idx]
        self._idx += 1
        return idx, problem, answer

    def __iter__(self):
        return self

class GSM8KMathsDataset(Dataset):
    def __init__(self, ds, seed: Optional[int] = None):
        if seed is not None:
            ds = ds.shuffle(seed=seed)
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
        problem = item["question"]
        answer = item["answer"]
        parsed_answer = parse_after_hashes(answer)
        return idx, problem, parsed_answer

    def __next__(self):
        if self._idx >= len(self):
            self._idx = 0
        idx, problem, answer = self[self._idx]
        self._idx += 1
        return idx, problem, answer

    def __iter__(self):
        return self
