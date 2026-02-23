import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available


def resolve_weight_paths(config: dict) -> tuple[str, str]:
    local_dir = config.get("weights_local_dir", "/tmp/llm-mcts-weights")
    value_path = os.path.join(local_dir, os.path.basename(config["value_head_path"]))
    policy_path = os.path.join(local_dir, os.path.basename(config["policy_head_path"]))
    return value_path, policy_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with the latest fine-tuned checkpoint"
    )
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional override for policy checkpoint directory.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional tokenizer source. Defaults to config model_name.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Physical GPU index to run on (sets CUDA_VISIBLE_DEVICES).",
    )
    return parser


def main():
    args = build_parser().parse_args()

    if args.rank is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.rank)

    with open(args.config, "r") as f:
        config = json.load(f)

    _, default_policy_path = resolve_weight_paths(config)
    checkpoint_path = args.checkpoint_path or default_policy_path
    tokenizer_path = args.tokenizer_path or config["model_name"]

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_path}\n"
            f"Expected a sharded HF directory from training publish."
        )

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    if attn_impl != "flash_attention_2":
        print("Warning: FlashAttention2 not available; using SDPA.")

    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Tokenizer source: {tokenizer_path}")
    print(f"Device: {device}, dtype: {dtype}, attn: {attn_impl}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)
    model.eval()

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print("Interactive chat ready. Commands: /reset, /quit")
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"/quit", "/exit"}:
            break
        if user_text.lower() == "/reset":
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            print("history reset")
            continue

        messages.append({"role": "user", "content": user_text})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        new_tokens = output_ids[0, input_ids.shape[1] :]
        assistant_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"assistant> {assistant_text}")
        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
