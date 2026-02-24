import json
import os

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from countdown_dataset import load_countdown_dataset, process_dataset_example
from eval import eval_countdown_hf, get_test_dataset
from graders import Graders

grader = Graders()


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


class PeriodicRewardEvalCallback(TrainerCallback):
    def __init__(self, config, tokenizer, eval_dataset):
        self.config = config
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.eval_every_steps = max(1, int(config.get("wandb_eval_every_steps", 24)))
        self.enabled = bool(config.get("wandb_eval_enabled", True))
        self.last_eval_step = -1

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self.enabled or model is None or self.eval_dataset is None:
            return control
        if not state.is_world_process_zero:
            return control
        if state.global_step == 0 or state.global_step % self.eval_every_steps != 0:
            return control
        if state.global_step == self.last_eval_step:
            return control

        try:
            score = eval_countdown_hf(
                model,
                self.eval_dataset,
                grader,
                self.config,
                tokenizer=self.tokenizer,
            )
            print(f"[baseline eval] step={state.global_step} eval/score={score:.4f}")
            if wandb.run is not None:
                wandb.log({"eval/score": score, "train/global_step": state.global_step})
            self.last_eval_step = state.global_step
        except Exception as exc:
            print(f"[baseline eval] step={state.global_step} eval failed: {exc}")
        return control


def main():
    config = json.load(open("configs/config.json"))
    dataset = load_countdown_dataset(seed=config.get("dataset_seed", 42))
    dataset = dataset.map(process_dataset_example)

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    eval_dataset = None
    if bool(config.get("baseline_eval_enabled", True)):
        eval_dataset = get_test_dataset(config)

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
        wandb.define_metric("train/global_step")
        wandb.define_metric("eval/*", step_metric="train/global_step")
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
        cast_lm_head_to_fp32=True,
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
    trainer.add_callback(PeriodicRewardEvalCallback(config, tokenizer, eval_dataset))

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
