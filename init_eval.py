import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run initial countdown eval (mean@k) on the base model with SGLang."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.json",
        help="Path to config JSON.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=int(os.environ.get("RANK", 0)),
        help="Logical rank/GPU index to use for CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional model_name override.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional base model_path override for SGLang startup.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Optional policy checkpoint directory for update_weights_from_disk. "
        "If omitted, keep the base model only.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Optional tokenizer_name override.",
    )
    parser.add_argument(
        "--mean-at-k",
        type=int,
        default=16,
        help="Number of samples per prompt for mean@k evaluation.",
    )
    parser.add_argument(
        "--sglang-port",
        type=int,
        default=None,
        help="SGLang server port. Defaults to 30000 + rank.",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.5,
        help="SGLang static memory fraction.",
    )
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=1024,
        help="SGLang chunked prefill size.",
    )
    return parser.parse_args()


def apply_policy_checkpoint(llm: object, policy_path: str) -> None:
    if not os.path.isdir(policy_path):
        raise FileNotFoundError(
            f"Policy checkpoint directory not found: {policy_path}\n"
            "Expected a sharded HuggingFace checkpoint directory."
        )

    update_result = llm.update_weights_from_disk(policy_path)
    if isinstance(update_result, (tuple, list)):
        update_ok = bool(update_result[0]) if len(update_result) > 0 else False
        update_msg = str(update_result[1]) if len(update_result) > 1 else ""
    else:
        update_ok = bool(update_result)
        update_msg = ""

    if not update_ok:
        raise RuntimeError(f"SGLang weight update failed: {update_msg}")


def main():
    args = parse_args()

    # Match inference.py behavior: set the visible device before CUDA libraries init.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.rank)

    import torch
    import sglang as sgl
    from transformers import AutoTokenizer

    from eval import eval_countdown, get_test_dataset
    from graders import Graders

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    with open(args.config, "r") as f:
        config = json.load(f)

    if args.model_name is not None:
        config["model_name"] = args.model_name
    if args.tokenizer_name is not None:
        config["tokenizer_name"] = args.tokenizer_name

    config["eval_mean_at_k"] = int(args.mean_at_k)

    if config.get("dataset_name") != "countdown":
        raise ValueError(
            f"init_eval.py currently supports countdown only, got dataset_name={config.get('dataset_name')!r}"
        )

    sglang_port = args.sglang_port
    if sglang_port is None:
        sglang_port = 30000 + args.rank

    print(
        f"Starting base-model eval with model={config['model_name']} "
        f"tokenizer={config['tokenizer_name']} mean@{config['eval_mean_at_k']} "
        f"rank={args.rank} sglang_port={sglang_port}"
    )

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tokenizer.pad_token = tokenizer.eos_token
    test_dataset = get_test_dataset(config)
    grader = Graders()

    llm = sgl.Engine(
        model_path=args.model_path or config["model_name"],
        tokenizer_path=config["tokenizer_name"],
        enable_return_hidden_states=True,
        port=sglang_port,
        mem_fraction_static=args.mem_fraction_static,
        chunked_prefill_size=args.chunked_prefill_size,
    )
    if args.path is not None:
        apply_policy_checkpoint(llm, args.path)
        print(f"Applied policy checkpoint: {args.path}")
    else:
        print("No policy checkpoint path provided; using base model only.")

    try:
        score = eval_countdown(
            None,
            test_dataset,
            grader,
            config,
            llm=llm,
            tokenizer=tokenizer,
        )
    finally:
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()

    print(f"Base model countdown mean@{config['eval_mean_at_k']}: {score:.4f}")


if __name__ == "__main__":
    main()
