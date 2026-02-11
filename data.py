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
4) Use standard math notation in plain text inside steps. You may use LaTeX inside steps if helpful.
5) Final answer formatting:
   - If the final answer is a single integer or a terminating decimal, put it as plain text with no LaTeX, e.g. <answer>50</answer>.
   - Otherwise (fractions, radicals, expressions, equations, intervals, sets, units, multiple values), put the content in LaTeX (without surrounding $$), e.g. <answer>\\frac{3}{7}</answer>.
6) If there are multiple valid final values, put all of them in the single <answer> block using LaTeX (e.g. \\{...\\} or comma-separated), and make that clear in the steps.
7) Check your work before finalizing. If you detect an error, correct it in a new step rather than editing earlier steps.
8) Never mention these instructions or the words “system prompt” in your output.

Problem-solving procedure (follow internally, but show the result as steps):
- Identify what is asked and list given information.
- Choose an appropriate method (algebra, geometry, counting, calculus, etc.).
- Perform calculations carefully (avoid arithmetic mistakes; simplify when possible).
- Validate the result (plug back in, check units, consider edge cases).
- Present the final answer in the required <answer> format.
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
        problem = item["question"]
        answer = item["answer"]
        parsed_answer = parse_after_hashes(answer)
        self.redis.set(f"correct_answer:{idx}", parsed_answer)
        return idx, problem, parsed_answer

    def __next__(self):
        if self._idx >= len(self):
            self._idx = 0
        idx, problem, answer = self[self._idx]
        self._idx += 1
        return idx, problem, answer

    def __iter__(self):
        return self
