from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
import torch
import os
from datasets import load_dataset
import json
import wandb
import re
from data import system_prompt

def extract_hash_answer(text):
    """Extract numerical answer from GSM8K format (#### marker)"""
    if "####" not in text:
        return None
    # GSM8K uses format: "Explanation... #### 42"
    return text.split("####")[1].strip()


def process_dataset_example(example):
    question = example["question"]
    answer = extract_hash_answer(example["answer"])
    
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    return {
        "prompt": prompt,
        "answer": answer,
    }


def extract_final_number(text: str):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I)
    if not m:
        return None
    return m.group(1).strip()

def reward_func(completions, answer=None, **kwargs):
    if answer is None:
        answer = kwargs["answer"]

    rewards = []
    for comp, gold in zip(completions, answer):
        text = comp[0]["content"] if isinstance(comp, list) else comp

        pred = extract_final_number(text)  # reads <answer>...</answer>
        if pred is None:
            rewards.append(-1.0)  # missing tags / no extracted answer
            continue

        # small format bonus for having an extracted answer
        pred_norm = pred.replace(",", "").strip()
        gold_norm = str(gold).replace(",", "").strip()

        if pred_norm == gold_norm:
            rewards.append(1.1)   # 0.1 format + 1.0 correct
        else:
            rewards.append(-0.9)  # 0.1 format - 1.0 incorrect

    return rewards



def main():
    config = json.load(open("configs/config.json"))
    dataset = load_dataset(config["dataset_name"], "main", split="train")
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
        wandb.init(project="mcts-language-model", name="grpo-baseline-qwen1.5b-gsm8k", config={
            "model_name": config["model_name"],
            "dataset_name": config["dataset_name"],
            "group_size": config["topk"],
            "per_device_train_batch_size": config["training_batch_size"] // 2,
        })
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