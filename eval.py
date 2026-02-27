import functools

import torch
from transformers import AutoTokenizer

from countdown_dataset import load_countdown_dataset, process_dataset_example
from tqdm import tqdm


def build_sample(sample, tokenizer):
    prompt = process_dataset_example(sample)["prompt"]
    input_numbers = sample.get("input_numbers")
    if input_numbers is None:
        input_numbers = sample.get("input")

    tokenized_prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "model_input": tokenized_prompt,
        "target": sample.get("target"),
        "input_numbers": input_numbers,
    }


def get_test_dataset(config):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    dataset = load_countdown_dataset(seed=config.get("dataset_seed", 42), split="test")
    dataset = dataset.map(functools.partial(build_sample, tokenizer=tokenizer))
    return dataset


def _generate_with_sglang(llm, tokenizer, prompts, config):
    temperature = float(config.get("eval_temperature", 0.2))
    sampling_params = {
        "max_new_tokens": int(config.get("eval_max_new_tokens", 1024)),
        "temperature": temperature,
    }
    if temperature > 0.0:
        sampling_params["top_p"] = float(config.get("sampling_top_p", 1.0))

    batch_size = int(
        config.get(
            "eval_generation_batch_size",
            config.get("inference_batch_size", 32),
        )
    )
    batch_size = max(1, batch_size)

    decoded_full_outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating with SGLang"):
        prompt_batch = prompts[i : i + batch_size]
        prompt_token_ids = [tokenizer.encode(prompt) for prompt in prompt_batch]
        outputs = llm.generate(
            input_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
        full_output_ids = [
            prompt_ids + output["output_ids"]
            for prompt_ids, output in zip(prompt_token_ids, outputs)
        ]
        decoded_full_outputs.extend(
            tokenizer.batch_decode(full_output_ids, skip_special_tokens=False)
        )
    return decoded_full_outputs


@torch.inference_mode()
def _generate_with_hf(model, tokenizer, prompts, config):
    temperature = float(config.get("eval_temperature", 0.2))
    batch_size = max(1, int(config.get("baseline_eval_batch_size", 8)))
    max_new_tokens = int(
        config.get(
            "baseline_eval_max_new_tokens", config.get("eval_max_new_tokens", 256)
        )
    )
    do_sample = temperature > 0.0
    top_p = float(config.get("sampling_top_p", 1.0))

    decoded_responses = []
    for i in range(0, len(prompts), batch_size):
        prompt_batch = prompts[i : i + batch_size]
        tokenized = tokenizer(prompt_batch, return_tensors="pt", padding=True).to(
            model.device
        )
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        output_ids = model.generate(**tokenized, **generation_kwargs)
        generated_ids = output_ids[:, tokenized["input_ids"].shape[1] :]
        decoded_responses.extend(
            tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        )
    return decoded_responses


def _generate_with_vllm(vllm_generation, tokenizer, prompts, config):
    temperature = float(config.get("eval_temperature", 0.2))
    batch_size = max(1, int(config.get("baseline_eval_batch_size", 32)))
    max_new_tokens = int(
        config.get(
            "baseline_eval_max_new_tokens", config.get("eval_max_new_tokens", 256)
        )
    )
    top_p = float(config.get("sampling_top_p", 1.0))

    old_temperature = vllm_generation.temperature
    old_top_p = vllm_generation.top_p
    old_max_completion_length = vllm_generation.max_completion_length

    decoded_responses = []
    try:
        vllm_generation.temperature = temperature
        vllm_generation.top_p = top_p if temperature > 0.0 else 1.0
        vllm_generation.max_completion_length = max_new_tokens

        for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating with VLLM"):
            prompt_batch = prompts[i : i + batch_size]
            _, completion_ids, _, _ = vllm_generation.generate(
                prompts=prompt_batch, num_generations=1
            )
            decoded_responses.extend(
                tokenizer.batch_decode(completion_ids, skip_special_tokens=False)
            )
    finally:
        vllm_generation.temperature = old_temperature
        vllm_generation.top_p = old_top_p
        vllm_generation.max_completion_length = old_max_completion_length

    return decoded_responses


def _group_eval_samples(test_dataset, config):
    k = int(config.get("eval_mean_at_k", 1))
    prompts = [sample["model_input"] for sample in test_dataset]
    prompts_eval_group = [prompt for prompt in prompts for _ in range(k)]
    samples_eval_group = [sample for sample in test_dataset for _ in range(k)]
    return prompts_eval_group, samples_eval_group


def _score_countdown_responses(decoded_responses, samples_eval_group, grader):
    if len(decoded_responses) != len(samples_eval_group):
        raise RuntimeError(
            f"eval response count mismatch: got {len(decoded_responses)} responses "
            f"for {len(samples_eval_group)} repeated samples"
        )

    rewards = []
    for response, sample in zip(decoded_responses, samples_eval_group):
        input_numbers = sample.get("input_numbers")
        if input_numbers is None:
            input_numbers = sample.get("input")
        reward = grader.countdown_reward_func(
            response,
            sample["target"],
            input_numbers,
            dense_rewards=False,
        )
        rewards.append(reward)
    return sum(rewards) / len(rewards)


def eval_countdown(_model, test_dataset, grader, config, llm=None, tokenizer=None):
    if llm is None:
        raise ValueError("eval_countdown requires an sglang llm instance")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    prompts_eval_group, samples_eval_group = _group_eval_samples(test_dataset, config)
    decoded_responses = _generate_with_sglang(
        llm, tokenizer, prompts_eval_group, config
    )
    return _score_countdown_responses(decoded_responses, samples_eval_group, grader)


def eval_countdown_hf(model, test_dataset, grader, config, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    max_samples = int(config.get("baseline_eval_num_samples", len(test_dataset)))
    max_samples = min(max_samples, len(test_dataset))
    eval_subset = [test_dataset[i] for i in range(max_samples)]

    was_training = model.training
    model.eval()
    try:
        prompts_eval_group, samples_eval_group = _group_eval_samples(
            eval_subset, config
        )
        decoded_responses = _generate_with_hf(
            model, tokenizer, prompts_eval_group, config
        )
        return _score_countdown_responses(decoded_responses, samples_eval_group, grader)
    finally:
        if was_training:
            model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def eval_countdown_vllm(
    model, test_dataset, grader, config, vllm_generation=None, tokenizer=None
):
    if vllm_generation is None:
        raise ValueError("eval_countdown_vllm requires a vllm_generation instance")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    max_samples = int(config.get("baseline_eval_num_samples", len(test_dataset)))
    max_samples = min(max_samples, len(test_dataset))
    eval_subset = [test_dataset[i] for i in range(max_samples)]

    was_training = model.training
    model.eval()
    try:
        vllm_generation.sync_weights()
        prompts_eval_group, samples_eval_group = _group_eval_samples(
            eval_subset, config
        )
        decoded_responses = _generate_with_vllm(
            vllm_generation, tokenizer, prompts_eval_group, config
        )
        return _score_countdown_responses(decoded_responses, samples_eval_group, grader)
    finally:
        if was_training:
            model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
