import argparse
import json
import os
import threading

import redis
import sglang as sgl
import wandb
from transformers import AutoTokenizer

from eval import _generate_with_sglang, _group_eval_samples, eval_countdown, get_test_dataset
from graders import Graders
from countdown_dataset import load_countdown_dataset, process_dataset_example
from weight_subscriber import run_weight_update_subscriber

grader = Graders()
N = 64


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run best-of-n data generation with optional startup weight load."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.json",
        help="Path to config JSON.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help=(
            "Optional checkpoint path for a one-time update_weights_from_disk call "
            "at startup, similar to init_eval.py."
        ),
    )
    parser.add_argument(
        "--outer-epochs",
        type=int,
        default=None,
        help=(
            "Optional override for outer epochs. If omitted, use "
            "best_of_n_outer_epochs from config."
        ),
    )
    return parser.parse_args()


def _resolve_policy_path(config):
    local_dir = config.get("weights_local_dir", "/tmp/llm-mcts-weights")
    return os.path.join(local_dir, os.path.basename(config["policy_head_path"]))


def _apply_policy_checkpoint(sglang, policy_path):
    update_result = sglang.update_weights_from_disk(policy_path)
    if isinstance(update_result, (tuple, list)):
        update_ok = bool(update_result[0]) if len(update_result) > 0 else False
        update_msg = str(update_result[1]) if len(update_result) > 1 else ""
    else:
        update_ok = bool(update_result)
        update_msg = ""
    if not update_ok:
        raise RuntimeError(f"SGLang weight update failed: {update_msg}")


def best_of_n(model, tokenizer, sample, config, N=N):
    sample_eval = {
        "model_input": tokenizer.apply_chat_template(
            sample["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        ),
        "target": sample["target"],
        "input_numbers": sample["input_numbers"],
    }
    prompt_text = sample_eval["model_input"]
    num_prompt_tokens = len(tokenizer.encode(prompt_text))

    prompts_eval_group, samples_eval_group = _group_eval_samples([sample_eval], config, N)
    best_of_n_temperature = float(
        config.get(
            "best_of_n_temperature",
            config.get("sampling_temperature", config.get("eval_temperature", 0.2)),
        )
    )
    decoded_full_outputs = _generate_with_sglang(
        model,
        tokenizer,
        prompts_eval_group,
        config,
        temperature=best_of_n_temperature,
    )

    scores = [
        grader.countdown_reward_func(
            response,
            sample_eval["target"],
            sample_eval["input_numbers"],
            dense_rewards=True,
        )
        for response in decoded_full_outputs
    ]

    best_of_n_output, best_of_n_score = max(zip(decoded_full_outputs, scores), key=lambda x: x[1])
    tokenized_best_of_n_output = tokenizer.encode(best_of_n_output)
    return num_prompt_tokens, tokenized_best_of_n_output, best_of_n_score


def best_of_n_loop(outer_epochs=None, startup_policy_path=None, config_path="configs/config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    if outer_epochs is None:
        outer_epochs = int(
            config.get("best_of_n_outer_epochs", config.get("outer_epochs", 1))
        )
    else:
        outer_epochs = int(outer_epochs)
    if outer_epochs < 1:
        raise ValueError(f"outer_epochs must be >= 1, got {outer_epochs}")
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must be in [0, {world_size - 1}], got {rank}")

    r = redis.Redis(
        host=config["redis_host"], port=config["redis_port"], db=config["redis_db"]
    )
    dataset = load_countdown_dataset(seed=config.get("dataset_seed", 42))
    dataset = dataset.map(process_dataset_example)
    dataset_total = len(dataset)
    shard_start = (dataset_total * rank) // world_size
    shard_end = (dataset_total * (rank + 1)) // world_size
    shard_indices = list(range(shard_start, shard_end))
    dataset = dataset.select(shard_indices)
    print(
        f"Rank {rank}: processing shard [{shard_start}:{shard_end}) "
        f"({len(dataset)} / {dataset_total} samples)"
    )

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    sglang_port = int(
        os.environ.get("SGLANG_PORT", config.get("sglang_port", 30000) + rank)
    )
    sglang = sgl.Engine(
        model_path=config["model_name"],
        tokenizer_path=config["tokenizer_name"],
        enable_return_hidden_states=True,
        port=sglang_port,
        mem_fraction_static=config.get("mem_fraction_static", 0.5),
        chunked_prefill_size=config.get("chunked_prefill_size", 1024),
    )
    policy_path = _resolve_policy_path(config)
    weights_state = {
        "last_version": 0,
        "primed": False,
    }
    eval_every_num_samples = int(config.get("best_of_n_eval_every_num_samples", 64))
    eval_owner_rank = int(config.get("best_of_n_eval_owner_rank", 2))
    eval_dataset = None
    eval_grader = None
    wandb_eval_enabled = bool(config.get("wandb_eval_enabled", True))
    if (
        rank == eval_owner_rank
        and config.get("dataset_name") == "countdown"
        and eval_every_num_samples > 0
    ):
        eval_dataset = get_test_dataset(config)
        eval_grader = Graders()
        print(
            f"Rank {rank}: eval enabled every {eval_every_num_samples} processed samples"
        )
        if wandb_eval_enabled:
            wandb.init(
                project=config.get("wandb_project", "mcts-language-model"),
                name=config.get(
                    "wandb_eval_run_name", f"best-of-n-eval-rank-{eval_owner_rank}"
                ),
                config={
                    "model_name": config.get("model_name"),
                    "dataset_name": config.get("dataset_name"),
                    "eval_every_num_samples": eval_every_num_samples,
                    "eval_owner_rank": eval_owner_rank,
                },
            )
    update_lock = threading.Lock()

    if startup_policy_path is not None:
        with update_lock:
            _apply_policy_checkpoint(sglang, startup_policy_path)
        weights_state["primed"] = True
        print(f"Applied startup policy checkpoint: {startup_policy_path}")

    def get_version():
        return weights_state["last_version"]

    def set_version(version):
        weights_state["last_version"] = int(version)
        if not weights_state["primed"] and int(version) > 0:
            weights_state["primed"] = True
            print(f"Primed policy weights to latest known version {version}")

    def sync_weights(version):
        with update_lock:
            _apply_policy_checkpoint(sglang, policy_path)

    def on_version_applied(previous_version: int, version: int):
        del previous_version
        if rank == eval_owner_rank:
            print(f"Rank {rank}: weights/version={version} applied")

    def on_error(error):
        print(f"Error syncing weights in best_of_n: {error}")

    threading.Thread(
        target=run_weight_update_subscriber,
        kwargs={
            "redis_host": config["redis_host"],
            "redis_port": config["redis_port"],
            "redis_db": config["redis_db"],
            "sync_weights": sync_weights,
            "get_current_version": get_version,
            "set_current_version": set_version,
            "on_version_applied": on_version_applied,
            "on_error": on_error,
        },
        daemon=True,
    ).start()

    samples_processed = 0
    shard_len = len(dataset)
    for outer_epoch in range(outer_epochs):
        print(f"Rank {rank}: starting outer epoch {outer_epoch + 1}/{outer_epochs}")
        epoch_offset = outer_epoch * shard_len
        for sample_index, sample in enumerate(dataset):
            with update_lock:
                (
                    num_prompt_tokens,
                    tokenized_best_of_n_output,
                    best_of_n_score,
                ) = best_of_n(sglang, tokenizer, sample, config, N)
            global_sample_index = shard_start + epoch_offset + sample_index
            prompt_id = sample.get("prompt_id", sample.get("id", global_sample_index))

            replay_buffer_entry = {
                "prompt_id": int(prompt_id),
                "state": tokenized_best_of_n_output,
                "reward": best_of_n_score,
                "num_prompt_tokens": num_prompt_tokens,
            }
            r.xadd("replay_buffer", {"data": json.dumps(replay_buffer_entry)})
            samples_processed += 1

            if (
                rank == eval_owner_rank
                and eval_dataset is not None
                and eval_every_num_samples > 0
                and samples_processed % eval_every_num_samples == 0
            ):
                with update_lock:
                    eval_score = eval_countdown(
                        None,
                        eval_dataset,
                        eval_grader,
                        config,
                        llm=sglang,
                        tokenizer=tokenizer,
                    )
                if wandb_eval_enabled:
                    wandb.log(
                        {
                            "eval/score": eval_score,
                            "weights/version": weights_state["last_version"],
                            "eval/samples_processed": samples_processed,
                        }
                    )
                print(
                    f"Rank {rank}: eval/score={eval_score:.4f} "
                    f"samples_processed={samples_processed} "
                    f"weights/version={weights_state['last_version']}"
                )

    r.close()


if __name__ == "__main__":
    args = parse_args()
    best_of_n_loop(
        outer_epochs=args.outer_epochs,
        startup_policy_path=args.path,
        config_path=args.config,
    )
