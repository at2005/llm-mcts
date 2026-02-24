import json

from datasets import Dataset

from data import countdown_system_prompt


def process_dataset_example(example):
    input_numbers = [int(n) for n in example["input"]]
    target = int(example["target"])
    question = f"Input Sequence: {input_numbers}\nTarget: {target}"

    prompt = [
        {"role": "system", "content": countdown_system_prompt},
        {"role": "user", "content": question},
    ]
    return {
        "prompt": prompt,
        "input_numbers": input_numbers,
        "target": target,
    }


def load_countdown_dataset(seed: int | None = None, split: str = "train") -> Dataset:
    rows = []
    fname = "envs/cd/dataset.json" if split == "train" else "envs/cd/dataset_test.json"
    with open(fname, "r") as f:
        for line in f:
            rows.append(json.loads(line))

    dataset = Dataset.from_list(rows)
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    return dataset
