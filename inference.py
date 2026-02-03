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
        self.value_head = ValueHead(self.hidden_size).to("cuda:0").eval()
        self.weights_lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Each rank needs its own sglang port to avoid conflicts
        sglang_port = 30000 + rank
        self.llm = sgl.Engine(model_path=model_name, enable_return_hidden_states=True, port=sglang_port)
        self.lm_head = nn.Linear(self.hidden_size, self.config["vocab_size"], bias=False).to(dtype=torch.bfloat16, device="cuda:0")

        self.rank = rank
        self.max_wait_ms = self.config["inference_max_wait_ms"]
        self.batch_size = self.config["inference_batch_size"]

        self.per_gpu_queue = queue.Queue()
    
        self.load_lm_head()
        self.stop_token_id = self.config["branch_token_id"]

    def load_lm_head(self):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda:0")
        lm_head_weight = model.lm_head.weight.detach().clone()
        self.lm_head.weight.data.copy_(lm_head_weight)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    def sync_weights(self):
        if os.path.exists(value_path) and os.path.exists(policy_path):
            with self.weights_lock:
                state_dict = torch.load(value_path, map_location="cuda:0", weights_only=True)
                self.value_head.load_state_dict(state_dict)
                self.llm.update_weights_from_disk(policy_path)
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
    def run_batch(self, input_ids):
        input_batch_text = [self.tokenizer.decode(input_id) for input_id in input_ids]
        print(f"Rank {self.rank}: Running batch with input_text: {input_batch_text}")
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": 1,
        }

        with self.weights_lock:
            # this is terrible performance but sglang is buggy for returning batched hidden states!!
            outputs = [self.llm.generate(input_ids=[ids], sampling_params=sampling_params, return_hidden_states=True, return_logprob=True, top_logprobs_num=self.config["topk"])[0] for ids in input_ids]

            policies = []
            last_hidden_states = []
            values = []

            for i, output in enumerate(outputs):
                meta_info = output["meta_info"]
                prefill_store = meta_info["hidden_states"][-1]

                print(f"Rank {self.rank}: Hidden States: {len(meta_info['hidden_states'])}")
                print(f"Rank {self.rank}: Prefill store length: {len(prefill_store)}")
                if len(prefill_store) == 0:
                    print(output)
                    print(self.tokenizer.decode(input_ids[i]))
                    raise ValueError("Prefill store is empty")

                if type(prefill_store[0]) == list:
                    prefill_store = prefill_store[-1]

                last_hidden_state = torch.tensor(prefill_store, dtype=torch.bfloat16).cuda(device="cuda:0") # [hidden_size]
                print(f"Rank {self.rank}: Last hidden state shape: {last_hidden_state.shape}")
                last_hidden_states.append(last_hidden_state)

            last_hidden_states = torch.stack(last_hidden_states)
            values : torch.Tensor = self.value_head(last_hidden_states.float()).squeeze(-1) # [batch_size]
            logits = self.lm_head(last_hidden_states)

        values = values.cpu().tolist()
        top_logits, top_indices = torch.topk(logits, self.config["topk"], dim=-1)

        top_probs = F.log_softmax(top_logits, dim=-1)

        top_probs = top_probs.cpu().tolist()
        top_indices = top_indices.cpu().tolist()

        policies = [{int(i): np.exp(p) for i, p in zip(indices, probs)} for indices, probs in zip(top_indices, top_probs)]

        return policies, values


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
                policy, value = self.run_batch(batch_input_ids)
                for fut, pol, val in zip(futures, policy, value):
                    fut.set_result((pol, val))
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
        policies, values = fut.result(timeout=3 * self.batch_inference_service.max_wait_ms / 1000.0)
        return inference_pb2.InferenceResponse(prior=policies, value=values)
    
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