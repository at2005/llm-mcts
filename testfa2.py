#!/usr/bin/env python3
import sys
import traceback


def main():
    import torch

    print("torch:", torch.__version__)

    import transformers

    print("transformers:", transformers.__version__)

    # ---- 1) Import + call flash-attn2 directly ----
    try:
        import flash_attn
        from flash_attn.flash_attn_interface import flash_attn_func

        print("flash_attn:", getattr(flash_attn, "__version__", "unknown"))
    except Exception as e:
        print("❌ Failed to import flash_attn / flash_attn_func:", repr(e))
        sys.exit(1)

    if not torch.cuda.is_available():
        print("❌ CUDA not available (flash-attn2 requires an NVIDIA GPU).")
        sys.exit(2)

    device = "cuda"
    torch.manual_seed(0)

    try:
        # q,k,v: (batch, seqlen, nheads, headdim)
        q = torch.randn(2, 16, 4, 64, device=device, dtype=torch.float16)
        k = torch.randn(2, 16, 4, 64, device=device, dtype=torch.float16)
        v = torch.randn(2, 16, 4, 64, device=device, dtype=torch.float16)
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        print(
            "✅ flash_attn_func OK:",
            tuple(out.shape),
            "finite:",
            bool(torch.isfinite(out).all()),
        )
    except Exception:
        print("❌ flash_attn_func call failed:")
        traceback.print_exc()
        sys.exit(3)

    # ---- 2) Force HF Transformers to use flash_attention_2 ----
    try:
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
        )

        # Newer Transformers: kwarg is accepted
        try:
            model = LlamaForCausalLM(cfg, attn_implementation="flash_attention_2")
        except TypeError:
            # Older Transformers: set on config
            cfg._attn_implementation = "flash_attention_2"
            model = LlamaForCausalLM(cfg)

        model.to(device=device, dtype=torch.float16).eval()

        input_ids = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
        with torch.no_grad():
            logits = model(input_ids).logits

        print(
            "✅ transformers flash_attention_2 forward OK:",
            tuple(logits.shape),
            "finite:",
            bool(torch.isfinite(logits).all()),
        )
    except Exception:
        print("❌ Transformers flash_attention_2 path failed:")
        traceback.print_exc()
        sys.exit(4)

    print("\n🎉 All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
