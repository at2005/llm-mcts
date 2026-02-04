import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"

class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor):
        return self.head(hidden_states)

class TrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.train()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.value_head = ValueHead(self.model.config.hidden_size)

    def forward(self, input_ids: torch.Tensor):
        outputs = self.model(input_ids, output_hidden_states=True)
        last_tok_hidden_states : torch.Tensor = outputs.last_hidden_state[-1][:, -1:, :] # [batch_size, hidden_size]

        logits : torch.Tensor = outputs.logits[:, -1, :] # [batch_size, vocab_size]
        value = self.value_head(last_tok_hidden_states).squeeze(-1) # [batch_size]

        return logits, value # [batch_size, vocab_size], [batch_size]