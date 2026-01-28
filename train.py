import torch
import torch.nn.functional as F
from model import TrainingModel
from redis import Redis
import time
import json

redis = Redis(host="localhost", port=6379, db=0)
max_steps = 100
train_batch_size = 128
c_value_loss = 1.0
c_policy_loss = 1.0

def train(rank: int):
    device = f"cuda:{rank}"
    model = TrainingModel().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    global_step = 0

    while global_step < max_steps:
        policy_batch = []
        value_batch = []
        state_batch = []
        
        while len(policy_batch) < train_batch_size:
            data = redis.lpop("replay_buffer")
            data = json.loads(data)
            if data is None:
                time.sleep(1)
                continue
            
            state = data["state"]
            policy = data["policy"]
            value = data["value"]

            policy_batch.append(policy)
            value_batch.append(value)
            state_batch.append(state)
    
        policy_batch = torch.tensor(policy_batch, dtype=torch.float32)
        value_batch = torch.tensor(value_batch, dtype=torch.bfloat16)
        state_batch = torch.tensor(state_batch, dtype=torch.long)

        policies, values = model(state_batch)

        value_loss = F.mse_loss(values, value_batch)
        policy_loss = F.kl_div(policies, policy_batch, reduction="batchmean")
        total_loss = c_value_loss * value_loss + c_policy_loss * policy_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1