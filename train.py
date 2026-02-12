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
        
        batch_tensor = model.tokenizer.pad({"input_ids" : state_batch}, padding=True, return_tensors="pt" )
        state_batch_tensor = batch_tensor["input_ids"].to(device)
        attention_mask = batch_tensor["attention_mask"].to(device)

        input_ids = state_batch_tensor[:, :-1]
        targets = state_batch_tensor[:, 1:]

        print(f"Rank {rank}: Input ids shape: {input_ids.shape}, Targets shape: {targets.shape}")

        logits, values = model(input_ids, attention_mask)
        print(f"Rank {rank}: Logits shape: {logits.shape}, Values shape: {values.shape}")
        value_loss = F.mse_loss(values.to(torch.float32), reward_batch_tensor.to(torch.float32))
        print(f"Rank {rank}: Value loss: {value_loss.item()}")

        ce_loss = F.cross_entropy(
            logits.to(torch.float32).view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id
        )
        print(f"Rank {rank}: Cross entropy loss: {ce_loss.item()}")

        total_loss: torch.Tensor = c_value_loss * value_loss + c_ce_loss * ce_loss
        print(f"Rank {rank}: Total loss: {total_loss.item()}")

        # wandb.log(
        #     {
        #         "loss/value_loss": value_loss.item(),
        #         "loss/ce_loss": ce_loss.item(),
        #         "loss/total_loss": total_loss.item(),
        #     }
        # )

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1


if __name__ == "__main__":
    train(7)