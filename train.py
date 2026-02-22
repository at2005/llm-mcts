import torch
import torch.nn as nn
import torch.nn.functional as F
from model import TrainingModel
from redis import Redis
from redis.exceptions import ResponseError
import time
import json
import os
import shutil
import wandb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
import functools
from tqdm import trange
import torch.distributed as dist

def resolve_weight_paths(config: dict) -> tuple[str, str]:
    local_dir = config.get("weights_local_dir", "/tmp/llm-mcts-weights")
    os.makedirs(local_dir, exist_ok=True)
    value_path = os.path.join(local_dir, os.path.basename(config["value_head_path"]))
    policy_path = os.path.join(local_dir, os.path.basename(config["policy_head_path"]))
    return value_path, policy_path


def ensure_replay_stream_group(redis: Redis, stream_key: str, group_name: str) -> None:
    """Idempotently ensure redis stream and consumer group are present."""
    try:
        redis.xgroup_create(stream_key, group_name, id="0", mkstream=True)
    except ResponseError as e:
        message = str(e).lower()
        if "busygroup" in message or "already exists" in message:
            return
        if "no such key" in message:
            redis.xgroup_create(stream_key, group_name, id="0", mkstream=True)
            return
        raise


def publish_weights(
    redis: Redis, model: FSDP, base_model: TrainingModel, config: dict
):
    value_path, policy_path = resolve_weight_paths(config)
    value_dir = os.path.dirname(os.path.abspath(value_path)) or "."
    tmp_value_path = os.path.join(
        value_dir, f".{os.path.basename(value_path)}.tmp"
    )
    policy_dir = os.path.abspath(policy_path)
    tmp_policy_dir = f"{policy_dir}.tmp"
    backup_policy_dir = f"{policy_dir}.bak"

    # Materialize full params directly from the FSDP-wrapped module for serialization.
    with FSDP.summon_full_params(model, recurse=True, writeback=False, rank0_only=True):
        if dist.get_rank() != 0:
            return

        # Extract value head weights and save.
        value_head_sd = {
            k: v.detach().cpu() for k, v in base_model.value_head.state_dict().items()
        }
        with open(tmp_value_path, "wb") as f:
            torch.save(value_head_sd, f)

        if os.path.isdir(tmp_policy_dir):
            shutil.rmtree(tmp_policy_dir)
        os.makedirs(tmp_policy_dir, exist_ok=True)
        base_model.model.save_pretrained(
            tmp_policy_dir, safe_serialization=True, max_shard_size="2GB"
        )

    os.replace(tmp_value_path, value_path)

    if os.path.isdir(backup_policy_dir):
        shutil.rmtree(backup_policy_dir)
    if os.path.isdir(policy_dir):
        os.rename(policy_dir, backup_policy_dir)
    elif os.path.isfile(policy_dir):
        os.remove(policy_dir)
    os.rename(tmp_policy_dir, policy_dir)
    if os.path.isdir(backup_policy_dir):
        shutil.rmtree(backup_policy_dir)

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


def build_kl_reference_model(
    config: dict, source_model: TrainingModel, device: str
) -> TrainingModel:
    kl_model = TrainingModel(config).to(device)
    kl_model.load_state_dict(source_model.state_dict(), strict=True)
    kl_model.model.gradient_checkpointing_disable()
    kl_model.model.config.use_cache = False
    kl_model.eval()
    for parameter in kl_model.parameters():
        parameter.requires_grad_(False)
    return kl_model


def ppo_step(
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    reward: torch.Tensor,
    attention_mask: torch.Tensor,
    generated_lengths: torch.Tensor,
    model: TrainingModel,
    kl_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    log_metrics: bool,
):
    epsilon = config["ppo_epsilon"]
    c_value_loss = config["c_value_loss"]
    c_policy_loss = config["c_policy_loss"]
    c_kl_loss = config["c_kl_loss"]
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

        selected_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        log_norm = torch.logsumexp(logits, dim=-1)
        log_probs_selected = selected_logits - log_norm

        # select for generated tokens, guaranteed to have no padding since we use left padding
        log_probs_selected = log_probs_selected.masked_fill(~logit_mask, 0)

        advantage: torch.Tensor = reward.unsqueeze(-1) - values.detach() # [B, T]
        advantage = advantage.masked_fill(~logit_mask, 0)

        kl_logits, _ = kl_model(input_ids, attention_mask)
        kl_selected_logits = kl_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        kl_log_norm = torch.logsumexp(kl_logits, dim=-1)
        kl_log_probs_selected = kl_selected_logits - kl_log_norm
        kl_log_probs_selected = kl_log_probs_selected.masked_fill(~logit_mask, 0)



    for _ in trange(num_ppo_inner_steps, desc="PPO inner steps", disable=not log_metrics):
        new_logits, new_values = model(input_ids, attention_mask)
        new_selected_logits = new_logits.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)  # [B, T]
        new_log_norm = torch.logsumexp(new_logits, dim=-1)
        new_log_probs_selected = new_selected_logits - new_log_norm

        # select for generated tokens, guaranteed to have no padding since we use left padding
        new_log_probs_selected = new_log_probs_selected.masked_fill(~logit_mask, 0)

        log_ratio = new_log_probs_selected - log_probs_selected
        log_ratio = torch.clamp(log_ratio, min=-10, max=10)
        ratio = torch.exp(log_ratio)
        ratio_clipped = ratio.clip(1 - epsilon, 1 + epsilon)  # [B, T]

        policy_loss: torch.Tensor = -torch.min(
            ratio * advantage, ratio_clipped * advantage
        )

        policy_loss = policy_loss.sum(dim=1) / denominator

        new_values = new_values.masked_fill(~logit_mask, 0)
        value_mse = (new_values - reward.unsqueeze(-1)) ** 2
        value_mse = value_mse.masked_fill(~logit_mask, 0)
        value_loss: torch.Tensor = value_mse.sum(dim=1) / denominator  # [B]
        kl_log_ratio = kl_log_probs_selected - new_log_probs_selected
        kl_loss = torch.exp(kl_log_ratio) - kl_log_ratio - 1.0
        kl_loss = kl_loss.masked_fill(~logit_mask, 0)
        kl_loss = kl_loss.sum(dim=1) / denominator

        value_loss = value_loss.mean()
        policy_loss = policy_loss.mean()
        kl_loss = kl_loss.mean()

        total_loss: torch.Tensor = (
            c_value_loss * value_loss + c_policy_loss * policy_loss + c_kl_loss * kl_loss
        )


        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

        if log_metrics:
            wandb.log(
                {
                    "loss/value_loss": value_loss.item(),
                    "loss/policy_loss": policy_loss.item(),
                    "loss/total_loss": total_loss.item(),
                    "loss/kl_loss": kl_loss.item(),
                    "reward/mean": reward.mean().item(),
                    "grad/norm": grad_norm.item(),
                }
            )
        optimizer.step()


def train(config: dict, redis: Redis, rank: int):
    device = f"cuda:{rank}"
    base_model = TrainingModel(config).to(device)
    tokenizer = base_model.tokenizer
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )

    kl_model = build_kl_reference_model(config, base_model, device)

    model = FSDP(base_model, auto_wrap_policy=wrap_policy, use_orig_params=True)
    model.train()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    max_steps = config["training_max_steps"]
    train_batch_size = config["training_batch_size"]
    max_wait_ms = config["training_max_wait_ms"]
    max_train_seqlen = config["max_train_seqlen"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], fused=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_rank0 = rank == 0
    global_step = 0
    if is_rank0:
        wandb.init(project="mcts-language-model", name="qwen1.5b-countdown", config=config)

    stream_key = "replay_buffer"
    group_name = "trainers"
    consumer_name = f"consumer_{rank}"
    ensure_replay_stream_group(redis, stream_key, group_name)
    dist.barrier()

    try:
        while global_step < max_steps:
            if (global_step + 1) % config["update_interval"] == 0:
                publish_weights(redis, model, base_model, config)

            reward_batch = []
            state_batch = []
            generated_lengths_batch = []

            while len(reward_batch) < train_batch_size:
                try:
                    resp = redis.xreadgroup(
                        group_name, consumer_name,
                        {stream_key: ">"},
                        count=1,
                        block=max_wait_ms,
                    )
                except ResponseError as e:
                    msg = str(e).lower()
                    if "no such key" in msg or "no such consumer group" in msg:
                        ensure_replay_stream_group(redis, stream_key, group_name)
                        continue
                    raise
                if not resp:
                    continue

                msg_id, fields = resp[0][1][0]
                redis.xack(stream_key, group_name, msg_id)
                data = json.loads(fields[b"data"])
                state = data["state"]
                reward = data["reward"]
                num_prompt_tokens = data["num_prompt_tokens"]
                num_generated_tokens = len(state) - num_prompt_tokens

                if len(state) > max_train_seqlen:
                    num_generated_tokens = max_train_seqlen - num_prompt_tokens
                    state = state[:max_train_seqlen]

                reward_batch.append(torch.tensor(reward, dtype=torch.bfloat16).to(device))
                state_batch.append(state)
                generated_lengths_batch.append(num_generated_tokens)

            reward_batch_tensor = torch.stack(reward_batch).to(device)
            generated_lengths_batch_tensor = torch.tensor(generated_lengths_batch).to(
                device
            )  # [B]

            batch_tensor = tokenizer.pad(
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
                kl_model,
                optimizer,
                config,
                log_metrics=is_rank0,
            )

            global_step += 1
    finally:
        if is_rank0:
            wandb.finish()


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    config = json.load(open("configs/config.json"))
    redis = Redis(
        host=config["redis_host"], port=config["redis_port"], db=config["redis_db"]
    )
    train(config, redis, dist.get_rank())
