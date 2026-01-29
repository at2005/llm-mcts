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
max_wait_ms = 10 * 1000

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
        
        deadline = time.monotonic() + (max_wait_ms / 1000.0)

        while len(policy_batch) < train_batch_size:

            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break

            data = redis.lpop("replay_buffer")
            if data is None:
                continue

            data = json.loads(data)

            state = data["state"]
            
            # dictionary
            priors = data["priors"]

            reward = data["reward"]

            policy_batch.append(priors)
            value_batch.append(reward)
            state_batch.append(state)

        value_batch_tensor = torch.tensor(value_batch, dtype=torch.bfloat16).to(device)
        state_batch_tensor = torch.tensor(state_batch, dtype=torch.long).to(device)

        probs = []
        actions = []
        batch_idx = []
        for b in range(len(policy_batch)):
            for action, prob in policy_batch[b]:
                probs.append(prob)
                actions.append(action)
                batch_idx.append(b)
        
        probs = torch.tensor(probs, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        batch_idx = torch.tensor(batch_idx, dtype=torch.long).to(device)

        policies, values = model(state_batch_tensor)

        selected_logprobs : torch.Tensor = policies[batch_idx, actions]
        policy_cross_entropy = - (probs * selected_logprobs).mean()

        value_loss = F.mse_loss(values, value_batch_tensor)
        total_loss = c_value_loss * value_loss + c_policy_loss * policy_cross_entropy 
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1