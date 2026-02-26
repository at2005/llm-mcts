import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available



class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, hidden_states: torch.Tensor):
        return self.head(hidden_states)


class TrainingModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model_name = self.config["model_name"]
        self.value_norm = nn.LayerNorm(config["hidden_size"]).to(torch.bfloat16)
        attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        if attn_impl != "flash_attention_2":
            print(
                "Warning: FlashAttention2 not available for training HF model; falling back to SDPA."
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self.model.train()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["tokenizer_name"])
        self.value_head = ValueHead(self.config["hidden_size"]).to(torch.bfloat16)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            base_outputs = self.model.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden_states = base_outputs.last_hidden_state

        logits = self.model.lm_head(hidden_states)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            value = self.value_head(self.value_norm(hidden_states)).squeeze(
                -1
            )  # [B, T]

        return logits, value  # [batch_size, vocab_size], [batch_size]
