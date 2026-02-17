import os

# Must set CUDA_VISIBLE_DEVICES before any CUDA imports
_rank = int(os.environ.get("RANK", 0))
os.environ["CUDA_VISIBLE_DEVICES"] = str(_rank)

import grpc
from concurrent import futures
import inference_pb2_grpc
import inference_pb2
from inference_pb2 import InferenceRequest, GraderRequest, GetPromptRequest
import torch
import threading
import sglang as sgl
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from model import ValueHead
from concurrent.futures import Future
from graders import Graders
import time
import queue
import re
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gc
from redis import Redis
import json
from data import get_dataset, system_prompt_message
from datasets import load_dataset
from transformers.modeling_utils import load_sharded_checkpoint
from safetensors.torch import load_file as load_safetensors_file


def resolve_weight_paths(config: dict) -> tuple[str, str]:
    local_dir = config.get("weights_local_dir", "/tmp/llm-mcts-weights")
    value_path = os.path.join(local_dir, os.path.basename(config["value_head_path"]))
    policy_path = os.path.join(local_dir, os.path.basename(config["policy_head_path"]))
    return value_path, policy_path


def load_policy_checkpoint(model: AutoModelForCausalLM, checkpoint_path: str, strict: bool) -> None:
    # support both sharded HF checkpoints and a single-file save_pretrained result
    safe_index = os.path.join(checkpoint_path, "model.safetensors.index.json")
    bin_index = os.path.join(checkpoint_path, "pytorch_model.bin.index.json")
    if os.path.isfile(safe_index) or os.path.isfile(bin_index):
        load_sharded_checkpoint(model, checkpoint_path, strict=strict, prefer_safe=True)
        return

    safe_file = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.isfile(safe_file):
        state_dict = load_safetensors_file(safe_file, device="cpu")
        model.load_state_dict(state_dict, strict=strict)
        return

    bin_file = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.isfile(bin_file):
        state_dict = torch.load(bin_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=strict)
        return

    raise FileNotFoundError(
        f"No supported checkpoint files found in {checkpoint_path}. "
        "Expected one of: model.safetensors.index.json, pytorch_model.bin.index.json, "
        "model.safetensors, pytorch_model.bin."
    )


class BatchInferenceService:
    def __init__(self, rank: int, config: dict):
        self.config = config

        # CUDA_VISIBLE_DEVICES is set at module load time, so cuda:0 refers to the assigned GPU
        self.hidden_size = self.config["hidden_size"]
        self.value_head = (
            ValueHead(self.hidden_size).to(device="cuda:0", dtype=torch.bfloat16).eval()
        )
        self.model_name = self.config["model_name"]
        self.weights_lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["tokenizer_name"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Each rank needs its own sglang port to avoid conflicts
        sglang_port = 30000 + rank
        self.llm = sgl.Engine(
            model_path=self.model_name,
            enable_return_hidden_states=True,
            port=sglang_port,
            mem_fraction_static=0.5,
            chunked_prefill_size=1024,
        )
        attn_impl = (
            "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        )
        if attn_impl != "flash_attention_2":
            print(
                f"Rank {rank}: Warning: FlashAttention2 not available for HF scorer model; falling back to SDPA."
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        ).to("cuda:0")
        self.model.config.use_cache = False
        self.model.eval()

        self.rank = rank
        self.max_wait_ms = self.config["inference_max_wait_ms"]
        self.batch_size = self.config["inference_batch_size"]

        self.per_gpu_queue = queue.Queue()

    def sync_weights(self):
        value_path, policy_path = resolve_weight_paths(self.config)
        if os.path.exists(value_path) and os.path.exists(policy_path):
            with self.weights_lock:
                state_dict = torch.load(
                    value_path, map_location="cuda:0", weights_only=True
                )
                self.value_head.load_state_dict(state_dict)
                update_result = self.llm.update_weights_from_disk(policy_path)
                if isinstance(update_result, (tuple, list)):
                    update_ok = bool(update_result[0]) if len(update_result) > 0 else False
                    update_msg = str(update_result[1]) if len(update_result) > 1 else ""
                else:
                    update_ok = bool(update_result)
                    update_msg = ""
                if not update_ok:
                    raise RuntimeError(
                        f"SGLang weight update failed: {update_msg}"
                    )
                if self.config["model_name"] == "meta-llama/Llama-3.2-1B-Instruct" or self.config["model_name"] == "Qwen/Qwen2.5-0.5B-Instruct":
                    load_policy_checkpoint(self.model, policy_path, strict=False)
                else:
                    load_policy_checkpoint(self.model, policy_path, strict=False)
                torch.cuda.empty_cache()
            # print(f"Rank {self.rank}: Loaded new weights")

    def weight_subscriber(self):
        redis = Redis(
            host=self.config["redis_host"],
            port=self.config["redis_port"],
            db=self.config["redis_db"],
        )
        pubsub = redis.pubsub()
        pubsub.subscribe("weights:updates")

        print(f"Rank {self.rank}: Subscribed to weights:updates")

        for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    json.loads(message["data"])
                    self.sync_weights()
                except Exception as e:
                    print(f"Rank {self.rank}: Error syncing weights: {e}")

    @torch.inference_mode()
    def compute_logprobs(self, input_ids, generated_ids):
        batch_token_ids = [inp + outp for inp, outp in zip(input_ids, generated_ids)]
        generated_lengths = [len(outp) for outp in generated_ids]

        max_length = max(len(token_ids) for token_ids in batch_token_ids)

        padded_input_ids = torch.full(
            (len(batch_token_ids), max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device="cuda:0",
        )
        attention_mask = torch.zeros(
            (len(batch_token_ids), max_length), dtype=torch.long, device="cuda:0"
        )

        for i, token_ids in enumerate(batch_token_ids):
            # left padding
            padded_input_ids[i, -len(token_ids) :] = torch.tensor(
                token_ids, dtype=torch.long, device="cuda:0"
            )
            attention_mask[i, -len(token_ids) :] = 1

        model_outputs = self.model.model(
            input_ids=padded_input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden_states = model_outputs.last_hidden_state

        summed_logprobs = torch.zeros(
            (len(batch_token_ids)), dtype=torch.float32, device="cuda:0"
        )
        hidden_states = torch.zeros(
            (len(batch_token_ids), self.hidden_size),
            dtype=torch.bfloat16,
            device="cuda:0",
        )

        for i in range(len(batch_token_ids)):
            gen_length = generated_lengths[i]
            hidden_states[i] = last_hidden_states[i, -gen_length - 1, :]

            labels = padded_input_ids[i, -gen_length:]
            logits_tok = self.model.lm_head(last_hidden_states[i, -gen_length - 1 : -1, :])
            selected_logits = logits_tok.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            log_norm = torch.logsumexp(logits_tok, dim=-1)
            summed_logprobs[i] = (selected_logits - log_norm).sum()

        return summed_logprobs, hidden_states

    @torch.inference_mode()
    def run_batch(self, input_ids):
        sampling_params = {
            "temperature": self.config["sampling_temperature"],
            "max_new_tokens": self.config["max_new_tokens"],
            "stop": self.config["stop_phrase"],
        }

        N = self.config["topk"]

        with self.weights_lock:
            replicated_input_ids = []

            for inp in input_ids:
                replicated_input_ids.extend([inp] * N)

            outputs = self.llm.generate(
                input_ids=replicated_input_ids, sampling_params=sampling_params
            )

            generated_ids = [output["output_ids"] for output in outputs]
            summed_logprobs, last_hidden_state = self.compute_logprobs(
                replicated_input_ids, generated_ids
            )

            # [batch_size * N] -> [batch_size, N]
            grouped_summed_logprobs = summed_logprobs.reshape(-1, N)  # [batch_size, N]

            # softmax across the N dimension
            grouped_summed_logprobs = F.softmax(grouped_summed_logprobs, dim=-1)

            grouped_generated_ids = [
                generated_ids[i : i + N] for i in range(0, len(generated_ids), N)
            ]  # [batch_size, N]

            values: torch.Tensor = self.value_head(last_hidden_state).squeeze(
                -1
            )  # [batch_size]
            grouped_values = values.reshape(-1, N).mean(dim=-1)  # [batch_size]

        return (
            grouped_generated_ids,
            grouped_summed_logprobs.cpu().tolist(),
            grouped_values.cpu().tolist(),
        )

    def batch_worker(self):
        while True:
            item = self.per_gpu_queue.get()
            batch = [item]
            deadline = time.monotonic() + (self.max_wait_ms / 1000.0)

            while len(batch) < self.batch_size:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break

                try:
                    item = self.per_gpu_queue.get(timeout=timeout)
                    batch.append(item)
                except queue.Empty:
                    break

            batch_input_ids = [item[0] for item in batch]
            futures = [item[1] for item in batch]
            # print(f"Rank {self.rank}: Running batch of size {len(batch)}")

            try:
                generated_ids, probabilities, values = self.run_batch(batch_input_ids)
                for fut, tok_ids, policies, val in zip(
                    futures, generated_ids, probabilities, values
                ):
                    policy = [
                        inference_pb2.PriorEntry(state=tok_ids, prior=prob)
                        for tok_ids, prob in zip(tok_ids, policies)
                    ]
                    fut.set_result((policy, val))
            except Exception as e:
                for fut in futures:
                    fut.set_exception(e)


class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self, batch_inference_service: BatchInferenceService):
        self.batch_inference_service = batch_inference_service
        self.graders = Graders()
        self.redis = Redis(
            host=self.batch_inference_service.config["redis_host"],
            port=self.batch_inference_service.config["redis_port"],
            db=self.batch_inference_service.config["redis_db"],
        )
        dataset_seed = self.batch_inference_service.config.get("dataset_seed")
        
        # different splits per rank but replicable across jobs 
        dataset_seed += self.batch_inference_service.rank

        self.dataset_name = self.batch_inference_service.config.get("dataset_name")
        self.maths_dataset = get_dataset(self.dataset_name, seed=dataset_seed)

    def infer(self, request: InferenceRequest, context):
        fut = Future()
        state = list(request.state)
        self.batch_inference_service.per_gpu_queue.put((state, fut))
        policies, values = fut.result(
            timeout=60 * self.batch_inference_service.max_wait_ms / 1000.0
        )
        return inference_pb2.InferenceResponse(priors=policies, value=values)

    def grader(self, request: GraderRequest, context):
        state = request.state
        prompt_id = request.prompt_id
        string_state = self.batch_inference_service.tokenizer.decode(state)
        if self.dataset_name == "openai/gsm8k":
            reward = self.graders.gsm8k_grader(string_state, prompt_id)
        elif self.dataset_name == "EleutherAI/hendrycks_math":
            reward = self.graders.maths_grader(string_state, prompt_id)
        return inference_pb2.GraderResponse(reward=reward)

    def get_prompt(self, request: GetPromptRequest, context):
        _, problem, answer = next(self.maths_dataset)
        prompt_uid = int(self.redis.incr("prompt_uid_counter"))
        answer_to_store = "" if answer is None else str(answer)
        self.redis.set(f"correct_answer:{prompt_uid}", answer_to_store)
        message = [system_prompt_message, {"role": "user", "content": problem}]
        chat_template = self.batch_inference_service.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized_ids = self.batch_inference_service.tokenizer.encode(chat_template)
        return inference_pb2.GetPromptResponse(prompt_id=prompt_uid, problem=tokenized_ids)


def test(rank: int):
    worker = BatchInferenceService(rank)
    messages = [
        {
            "role": "user",
            "content": "Complete the sentence with just one word: The capital of France is: ",
        }
    ]
    test_state = worker.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    test_state = worker.tokenizer.encode(test_state)

    test_policy, test_value = worker.run_batch([test_state])
    print(f"Rank {rank}: {test_policy}")
    print(f"Rank {rank}: {test_value}")
    max_token = max(test_policy[0], key=test_policy[0].get)
    print(f"Rank {rank}: {max_token}")
    decoded = worker.tokenizer.decode([max_token])
    print(f"Rank {rank}: {decoded}")


def serve():
    with open("configs/config.json", "r") as f:
        config = json.load(f)

    rank = int(os.environ.get("RANK", 0))
    port = int(os.environ.get("PORT", config["inference_base_port"] + rank))
    print(f"Rank {rank}: Starting server on port {port}")

    worker = BatchInferenceService(rank, config)
    threading.Thread(target=worker.batch_worker, daemon=True).start()
    threading.Thread(target=worker.weight_subscriber, daemon=True).start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1024))
    inference_pb2_grpc.add_InferenceServicer_to_server(
        InferenceServicer(worker), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Server started on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
