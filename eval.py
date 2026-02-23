import functools
from transformers import AutoTokenizer

from baseline_countdown import load_countdown_dataset, process_dataset_example


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
            config.get("inference_batch_size", 8),
        )
    )
    batch_size = max(1, batch_size)

    decoded_full_outputs = []
    for i in range(0, len(prompts), batch_size):
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


def eval_countdown(_model, test_dataset, grader, config, llm=None, tokenizer=None):
    if llm is None:
        raise ValueError("eval_countdown requires an sglang llm instance")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    prompts = [sample["model_input"] for sample in test_dataset]
    prompts_eval_group = [
        prompt for prompt in prompts for _ in range(config.get("eval_mean_at_k", 1))
    ]
    decoded_responses = _generate_with_sglang(
        llm, tokenizer, prompts_eval_group, config
    )

    rewards = []
    for response, sample in zip(decoded_responses, test_dataset):
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
