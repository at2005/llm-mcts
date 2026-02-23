import functools
import torch
from transformers import AutoTokenizer

from baseline_countdown import load_countdown_dataset, process_dataset_example

def build_sample(sample, tokenizer):
    prompt = process_dataset_example(sample)["prompt"]

    tokenized_prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"model_input": tokenized_prompt}

def get_test_dataset(config):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    dataset = load_countdown_dataset(seed=config.get("dataset_seed", 42))
    dataset = dataset.map(functools.partial(build_sample, tokenizer=tokenizer))
    return dataset


def eval_countdown(model, test_dataset, grader, config):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    prompts = [sample["model_input"] for sample in test_dataset]
    model_inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    temperature = float(config.get("eval_temperature", 0.0))
    do_sample = temperature > 0.0
    generate_kwargs = dict(
        max_new_tokens=1024,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if do_sample:
        generate_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=float(config.get("sampling_top_p", 1.0)),
        )
    else:
        generate_kwargs.update(do_sample=False)

    with torch.inference_mode():
        outputs = model.generate(**model_inputs, **generate_kwargs)

    generated_ids = outputs[:, model_inputs["input_ids"].shape[1] :]
    decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    rewards = []
    for response, sample in zip(decoded_responses, test_dataset):
        reward = grader.countdown_reward_func(response, sample["target"], sample["input_numbers"], dense_rewards=True)
        rewards.append(reward)
    return sum(rewards) / len(rewards)
