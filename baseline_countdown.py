from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
import torch
import os
from datasets import Dataset
import json
import wandb
from data import countdown_system_prompt
from graders import Graders


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


grader = Graders()


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


def reward_func(completions, target=None, input_numbers=None, **kwargs):
    if target is None:
        target = kwargs["target"]
    if input_numbers is None:
        input_numbers = kwargs["input_numbers"]

    rewards = []
    for comp, gold_target, numbers in zip(completions, target, input_numbers):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        reward = grader.countdown_reward_func(
            text, int(gold_target), [int(n) for n in numbers]
        )
        rewards.append(reward)
    return rewards


def main():
    config = json.load(open("configs/config.json"))
    dataset = load_countdown_dataset(seed=config.get("dataset_seed", 42))
    dataset = dataset.map(process_dataset_example)

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        **model_kwargs,
    )

    model.train()

    if int(os.environ.get("RANK", "0")) == 0:
        wandb.init(
            project="mcts-language-model",
            name="grpo-baseline-qwen1.5b-countdown",
            config={
                "model_name": config["model_name"],
                "dataset_name": config["dataset_name"],
                "group_size": config["topk"],
                "per_device_train_batch_size": config["training_batch_size"] // 2,
            },
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"

    training_args = GRPOConfig(
        learning_rate=5e-6,
        max_steps=500,
        logging_steps=1,
        per_device_train_batch_size=config["training_batch_size"] // 2,
        num_generations=config["topk"] * 4,
        # max_prompt_length=config["max_train_seqlen"],
        bf16=True,
        report_to="wandb",
        remove_unused_columns=False,
        max_completion_length=1024,
        max_grad_norm=0.1,
        chat_template_kwargs={
            "truncation": True,
            "max_length": 1024,
        },
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[reward_func],
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
