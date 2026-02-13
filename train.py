import torch
import torch.nn.functional as F
from model import TrainingModel
from redis import Redis
import time
import json
import os
import wandb


def publish_weights(
    redis: Redis, model: TrainingModel, config: dict
):
    value_path = config['value_head_path']
    policy_path = config['policy_head_path']
    tmp_value_path = f"/tmp/{value_path}"
    tmp_policy_path = f"/tmp/{policy_path}"

    with open(tmp_value_path, "wb") as f:
        torch.save(model.value_head.state_dict(), f)
    with open(tmp_policy_path, "wb") as f:
        torch.save(model.model.state_dict(), f)

    os.rename(tmp_value_path, value_path)
    os.rename(tmp_policy_path, policy_path)

    version = int(redis.incr("weights:version_counter"))
    meta = {
        "version": version,
        "value_head": value_path,
        "llm": policy_path,
        "ts": time.time(),
    }

    pipe = redis.pipeline()
    pipe.set(f"weights:meta:{version}", json.dumps(meta))
    pipe.set("weights:latest_version", version)
    pipe.execute()

    redis.publish("weights:updates", json.dumps({"version": version}))


def ppo_step(
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    reward: torch.Tensor,
    attention_mask: torch.Tensor,
    generated_lengths: torch.Tensor,
    model: TrainingModel,
    optimizer: torch.optim.Optimizer,
    config: dict,
):
    epsilon = config["ppo_epsilon"]
    c_value_loss = config["c_value_loss"]
    c_policy_loss = config["c_policy_loss"]
    num_ppo_inner_steps = config["num_ppo_inner_steps"]

    with torch.no_grad():
        logits, values = model(input_ids, attention_mask)

        # remove the last logit because we do not sample from it, ie no targets for it
        T = logits.shape[1]
        logit_mask = torch.arange(
            T, device=logits.device
        ) >= T - generated_lengths.unsqueeze(
            -1
        )  # [B, T]
        denominator = logit_mask.sum(dim=1).clamp(min=1)  # [B]

        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_selected = log_probs.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)

        # select for generated tokens, guaranteed to have no padding since we use left padding
        log_probs_selected = log_probs_selected.masked_fill(~logit_mask, 0)

        advantage: torch.Tensor = reward.unsqueeze(-1) - values  # [B, T]
        advantage = advantage.masked_fill(~logit_mask, 0)

    for _ in range(num_ppo_inner_steps):
        new_logits, new_values = model(input_ids, attention_mask)
        new_log_probs = F.log_softmax(new_logits, dim=-1)

        new_log_probs_selected = new_log_probs.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(
            -1
        )  # [B, T]

        # select for generated tokens, guaranteed to have no padding since we use left padding
        new_log_probs_selected = new_log_probs_selected.masked_fill(~logit_mask, 0)

        log_ratio = new_log_probs_selected - log_probs_selected
        ratio = torch.exp(log_ratio)
        ratio_clipped = ratio.clip(1 - epsilon, 1 + epsilon)  # [B, T]

        policy_loss: torch.Tensor = -torch.min(
            ratio * advantage, ratio_clipped * advantage
        )

        policy_loss = policy_loss.sum(dim=1) / denominator

        new_values = new_values.masked_fill(~logit_mask, 0)
        value_mse = (new_values - reward.unsqueeze(-1)) ** 2
        value_loss: torch.Tensor = value_mse.sum(dim=1) / denominator  # [B]

        total_loss: torch.Tensor = (
            c_value_loss * value_loss.mean() + c_policy_loss * policy_loss.mean()
        )

        wandb.log(
            {
                "loss/value_loss": value_loss.item(),
                "loss/policy_loss": policy_loss.item(),
                "loss/total_loss": total_loss.item(),
            }
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


def train(config: dict, redis: Redis, rank: int):
    device = f"cuda:{rank}"
    model = TrainingModel(config).to(device)
    model.train()
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"

    max_steps = config["training_max_steps"]
    train_batch_size = config["training_batch_size"]
    max_wait_ms = config["training_max_wait_ms"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    global_step = 0
    # wandb.init(project="mcts-language-model", name=f"train-rank-{rank}", config=config)

    while global_step < max_steps:
        if rank == 0 and (global_step + 1) % 10 == 0:
            publish_weights(redis, model, config)

        reward_batch = []
        state_batch = []
        generated_lengths_batch = []

        deadline = time.monotonic() + (max_wait_ms / 1000.0)

        while len(reward_batch) < train_batch_size:
            timeout = deadline - time.monotonic()
            if timeout <= 0 and len(reward_batch) > 0:
                break

            print(f"Rank {rank}: Waiting for data from replay buffer")
            data = redis.lpop("replay_buffer")
            if data is None:
                continue

            data = json.loads(data)
            print(f"Rank {rank}: Received data from replay buffer")
            state = data["state"]
            reward = data["reward"]
            num_prompt_tokens = data["num_prompt_tokens"]
            num_generated_tokens = len(state) - num_prompt_tokens

            print(
                f"Rank {rank}: Reward: {reward}, State: {state}, Num prompt tokens: {num_prompt_tokens}"
            )
            reward_batch.append(torch.tensor(reward, dtype=torch.bfloat16).to(device))
            state_batch.append(state)
            generated_lengths_batch.append(num_generated_tokens)

        reward_batch_tensor = torch.stack(reward_batch).to(device)
        generated_lengths_batch_tensor = torch.tensor(generated_lengths_batch).to(
            device
        )  # [B]

        batch_tensor = model.tokenizer.pad(
            {"input_ids": state_batch}, padding=True, return_tensors="pt"
        )
        state_batch_tensor = batch_tensor["input_ids"].to(device)
        attention_mask = batch_tensor["attention_mask"].to(device)

        input_ids = state_batch_tensor[:, :-1]
        targets = state_batch_tensor[:, 1:]
        attention_mask = attention_mask[:, :-1]

        ppo_step(
            input_ids,
            targets,
            reward_batch_tensor,
            attention_mask,
            generated_lengths_batch_tensor,
            model,
            optimizer,
            config,
        )

        global_step += 1


if __name__ == "__main__":
    config = json.load(open("configs/config.json"))
    redis = Redis(
        host=config["redis_host"], port=config["redis_port"], db=config["redis_db"]
    )
    train(config, redis, 5)
