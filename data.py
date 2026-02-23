from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from redis import Redis
import re
from typing import Optional
from transformers import AutoTokenizer
import json
import random

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
4) If the final answer is a single integer or a terminating decimal, put it as plain text with no LaTeX, e.g. <answer>50</answer>. Do not add dollar signs or any other units of measurement.
"""
countdown_system_prompt = """You are a mathematics reasoning assistant that solves Countdown number puzzles.

You are given a list of input numbers and a target number. Your goal is to combine the input numbers using +, -, *, / to reach the target. Each input number may only be used at most once. Intermediate results must be positive integers (no fractions).

OUTPUT FORMAT (strict):
- Produce one or more reasoning steps. Each step MUST be wrapped exactly like:
<|start_step|>...<|end_step|>
- After the final step, output the final arithmetic expression exactly once, wrapped like:
<answer>...</answer>

Rules:
1) Every message you produce must follow the format above: steps first, then a single <answer> block.
2) Do NOT put anything outside step blocks except the final <answer>...</answer>.
3) Each step should reason about how to reach the target and what operations to try. Think about what numbers are available and what the target requires.
4) The <answer> block should contain a single arithmetic expression using only the input numbers and +, -, *, / operators, e.g. <answer>(11 - 3) / 4 + 9</answer>.
"""


system_prompt_message = {"role": "system", "content": system_prompt}
countdown_system_prompt_message = {"role": "system", "content": countdown_system_prompt}


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
        return GSM8KMathsDataset(
            load_dataset(dataset_name, "main", split="train"), seed=seed
        )
    elif dataset_name == "EleutherAI/hendrycks_math":
        return MATHDataset(
            load_dataset(dataset_name, "algebra", split="train"), seed=seed
        )
    elif dataset_name == "countdown":
        return CountdownDataset(seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


class CountdownDataset(Dataset):
    def __init__(self, seed: Optional[int] = None):
        self.ds = []
        self._redis = None
        self._idx = 0
        with open("envs/cd/dataset.json", "r") as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                self.ds.append(data)

        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self.ds)

    @property
    def redis(self):
        if self._redis is None:
            self._redis = Redis(host="localhost", port=6379, db=0)
        return self._redis

    def _build_problem(self, in_seq: list[int], target: int) -> str:
        return f"Input Sequence: {in_seq}\nTarget: {target}"

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        in_seq = item["input"]
        target = item["target"]
        problem = self._build_problem(in_seq, target)
        return idx, problem, target, in_seq

    def __next__(self):
        if self._idx >= len(self):
            self._idx = 0
        idx, problem, target, in_seq = self[self._idx]
        self._idx += 1
        return idx, problem, target, in_seq

    def __iter__(self):
        return self


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
