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
from model import ValueHead, model_name
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
from data import MathsDataset, system_prompt_message
from datasets import load_dataset
from train import value_path, policy_path


class BatchInferenceService:
    def __init__(self, rank: int, config: dict):
        self.config = config

        # CUDA_VISIBLE_DEVICES is set at module load time, so cuda:0 refers to the assigned GPU
        self.hidden_size = self.config["hidden_size"]
        self.value_head = ValueHead(self.hidden_size).to(device="cuda:0", dtype=torch.bfloat16).eval()
        self.weights_lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Each rank needs its own sglang port to avoid conflicts
        sglang_port = 30000 + rank
        self.llm = sgl.Engine(model_path=model_name, enable_return_hidden_states=True, port=sglang_port, mem_fraction_static=0.2)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda:0")

        self.rank = rank
        self.max_wait_ms = self.config["inference_max_wait_ms"]
        self.batch_size = self.config["inference_batch_size"]

        self.per_gpu_queue = queue.Queue()
    
        self.stop_token_id = self.config["branch_token_id"]

    def sync_weights(self):
        if os.path.exists(value_path) and os.path.exists(policy_path):
            with self.weights_lock:
                state_dict = torch.load(value_path, map_location="cuda:0", weights_only=True)
                self.value_head.load_state_dict(state_dict)
                self.llm.update_weights_from_disk(policy_path)
                self.model.load_state_dict(torch.load(policy_path, map_location="cuda:0", weights_only=True))
            print(f"Rank {self.rank}: Loaded new weights")

    def weight_subscriber(self):
        redis = Redis(host=self.config["redis_host"], port=self.config["redis_port"], db=self.config["redis_db"])
        pubsub = redis.pubsub()
        pubsub.subscribe("weights:updates")

        print(f"Rank {self.rank}: Subscribed to weights:updates")

        for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    version = data.get("version")
                    print(f"Rank {self.rank}: Received weight update v{version}")
                    self.sync_weights()
                except Exception as e:
                    print(f"Rank {self.rank}: Error syncing weights: {e}")

    @torch.inference_mode()
    def compute_logprobs(self, input_ids, generated_ids):
        batch_token_ids = [inp + outp for inp, outp in zip(input_ids, generated_ids)]
        generated_lengths = [len(outp) for outp in generated_ids]

        max_length = max(len(token_ids) for token_ids in batch_token_ids)

        padded_input_ids = torch.full((len(batch_token_ids), max_length), self.tokenizer.pad_token_id, dtype=torch.long, device="cuda:0")
        attention_mask = torch.zeros((len(batch_token_ids), max_length), dtype=torch.long, device="cuda:0")

        for i, token_ids in enumerate(batch_token_ids):
            # left padding
            padded_input_ids[i, - len(token_ids) : ] = torch.tensor(token_ids, dtype=torch.long, device="cuda:0")
            attention_mask[i, - len(token_ids) : ] = 1

        model_outputs = self.model(input_ids=padded_input_ids, attention_mask=attention_mask, output_hidden_states=True)
        full_logits = model_outputs.logits
        last_hidden_states = model_outputs.hidden_states[-1]

        logprobs = F.log_softmax(full_logits, dim=-1) # [batch_size, max_length, vocab_size]
        
        summed_logprobs = torch.zeros((len(batch_token_ids)), dtype=torch.float32, device="cuda:0")
        hidden_states = torch.zeros((len(batch_token_ids), self.hidden_size), dtype=torch.bfloat16, device="cuda:0")

        for i in range(len(batch_token_ids)):
            gen_length = generated_lengths[i]
            hidden_states[i] = last_hidden_states[i, - gen_length - 1, :]

            labels = padded_input_ids[i, - gen_length: ]
            logprobs_tok = logprobs[i, - gen_length - 1: -1, :]
            summed_logprobs[i] = logprobs_tok.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).sum()

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

            outputs = self.llm.generate(input_ids=replicated_input_ids, sampling_params=sampling_params)

            generated_ids = [output["output_ids"] for output in outputs]
            summed_logprobs, last_hidden_state = self.compute_logprobs(replicated_input_ids, generated_ids)

            # [batch_size * N] -> [batch_size, N]
            grouped_summed_logprobs = summed_logprobs.reshape(-1, N) # [batch_size, N]

            grouped_generated_ids = [generated_ids[i:i+N] for i in range(0, len(generated_ids), N)] # [batch_size, N]

            values : torch.Tensor = self.value_head(last_hidden_state).squeeze(-1) # [batch_size]
            grouped_values = values.reshape(-1, N).mean(dim=-1) # [batch_size]

        return (grouped_generated_ids, grouped_summed_logprobs.exp().cpu().tolist(), grouped_values.cpu().tolist())


    def batch_worker(self):
        while True:
            item = self.per_gpu_queue.get()
            batch = [item]
            deadline = time.monotonic() + (self.max_wait_ms / 1000.0)

            while len(batch) < self.batch_size:
                print(f"Rank {self.rank}: Waiting for batch with length {len(batch)}")
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    print(f"Rank {self.rank}: Timeout waiting for batch")
                    break

                try:
                    item = self.per_gpu_queue.get(timeout=timeout)
                    batch.append(item)
                except queue.Empty:
                    break
            
            batch_input_ids = [item[0] for item in batch]
            futures = [item[1] for item in batch]

            try:
                print(f"Rank {self.rank}: Running batch with length {len(batch_input_ids)}")
                generated_ids, probabilities, values = self.run_batch(batch_input_ids)
                for fut, tok_ids, policies, val in zip(futures, generated_ids, probabilities, values):
                    policy = [inference_pb2.PriorEntry(state=tok_ids, prior=prob) for tok_ids, prob in zip(tok_ids, policies)]
                    fut.set_result((policy, val))
            except Exception as e:
                for fut in futures:
                    fut.set_exception(e)
            

class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self, batch_inference_service: BatchInferenceService):
        self.batch_inference_service = batch_inference_service
        self.graders = Graders()
        self.maths_dataset = MathsDataset(load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train"))

    def infer(self, request : InferenceRequest, context):
        fut = Future()
        state = list(request.state)
        self.batch_inference_service.per_gpu_queue.put((state, fut))
        policies, values = fut.result(timeout=60 * self.batch_inference_service.max_wait_ms / 1000.0)
        return inference_pb2.InferenceResponse(priors=policies, value=values)
    
    def grader(self, request : GraderRequest, context):
        state = request.state
        prompt_id = request.prompt_id
        string_state = self.batch_inference_service.tokenizer.decode(state)
        # parse answer in <answer>...</answer>
        reward = self.graders.maths_grader(string_state, prompt_id)
        return inference_pb2.GraderResponse(reward=reward)
    
    def get_prompt(self, request : GetPromptRequest, context):
        idx, problem, answer = next(self.maths_dataset)
        message = [system_prompt_message, {"role": "user", "content": problem}]
        chat_template = self.batch_inference_service.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True,)
        tokenized_ids = self.batch_inference_service.tokenizer.encode(chat_template)
        return inference_pb2.GetPromptResponse(prompt_id=idx, problem=tokenized_ids)


def test(rank: int):
    worker = BatchInferenceService(rank)
    messages = [{"role": "user", "content": "Complete the sentence with just one word: The capital of France is: "}]
    test_state = worker.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(worker), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f'Server started on port {port}')
    server.wait_for_termination()

if __name__ == "__main__":
    serve()