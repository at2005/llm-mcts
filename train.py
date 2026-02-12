import torch
import torch.nn.functional as F
from model import TrainingModel
from redis import Redis
import time
import json
import os
from torch.nn.utils.rnn import pad_sequence
import wandb

redis = Redis(host="localhost", port=6379, db=0)
config = json.load(open("configs/config.json"))

max_steps = config["training_max_steps"]
train_batch_size = config["training_batch_size"]
c_value_loss = config["c_value_loss"]
c_ce_loss = config["c_ce_loss"]
max_wait_ms = config["training_max_wait_ms"]

value_path = "value_head.pth"
policy_path = "llm.pth"


def publish_weights(model: TrainingModel):
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
        assert logits.dtype == torch.float32
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_selected = log_probs.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)

        bool_mask = attention_mask.bool()
        log_probs_selected = log_probs_selected * bool_mask
        values_masked = values * bool_mask
        advantage: torch.Tensor = reward.unsqueeze(-1) - values_masked  # [B, T]
        denominator = bool_mask.sum(dim=1).clamp(min=1)  # [B]

    for _ in range(num_ppo_inner_steps):
        new_logits, new_values = model(input_ids, attention_mask)
        new_log_probs = F.log_softmax(new_logits.to(torch.float32), dim=-1)
        new_log_probs_selected = new_log_probs.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(
            -1
        )  # [B, T]

        new_log_probs_selected = new_log_probs_selected * bool_mask

        log_ratio = new_log_probs_selected - log_probs_selected
        ratio = torch.exp(log_ratio)
        ratio_clipped = ratio.clip(1 - epsilon, 1 + epsilon)  # [B, T]

        policy_loss: torch.Tensor = -torch.min(
            ratio * advantage, ratio_clipped * advantage
        )

        policy_loss = (policy_loss * bool_mask).sum(dim=1) / denominator  # [B]

        value_mse = (new_values - reward.unsqueeze(-1)) ** 2
        value_loss = (value_mse * bool_mask).sum(dim=1) / denominator  # [B]

        total_loss: torch.Tensor = (
            c_value_loss * value_loss + c_policy_loss * policy_loss
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


def train(rank: int):
    device = f"cuda:{rank}"
    model = TrainingModel(config).to(device)
    model.train()
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    pad_id = model.tokenizer.pad_token_id
    global_step = 0
    # wandb.init(project="mcts-language-model", name=f"train-rank-{rank}", config=config)

    while global_step < max_steps:
        if rank == 0 and (global_step + 1) % 10 == 0:
            publish_weights(model)

        reward_batch = []
        state_batch = []

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

            print(f"Rank {rank}: Reward: {reward}, State: {state}")
            reward_batch.append(torch.tensor(reward, dtype=torch.bfloat16).to(device))
            state_batch.append(state)

        reward_batch_tensor = torch.stack(reward_batch).to(device)

        batch_tensor = model.tokenizer.pad(
            {"input_ids": state_batch}, padding=True, return_tensors="pt"
        )
        state_batch_tensor = batch_tensor["input_ids"].to(device)
        attention_mask = batch_tensor["attention_mask"].to(device)

        input_ids = state_batch_tensor[:, :-1]
        targets = state_batch_tensor[:, 1:]

        ppo_step(
            input_ids,
            targets,
            reward_batch_tensor,
            attention_mask,
            model,
            optimizer,
            config,
        )

        global_step += 1


if __name__ == "__main__":
    train(7)
