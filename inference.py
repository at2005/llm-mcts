import grpc
from concurrent import futures
import inference_pb2_grpc
import inference_pb2
from inference_pb2 import InferenceRequest, GraderRequest
import torch
import threading
import sglang as sgl
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import ValueHead, model_name
from concurrent.futures import Future
from graders import Graders
import time
import queue
import os
import re
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gc
from redis import Redis
import json

from train import value_path, policy_path

topk = 4
hidden_size = 4096
vocab_size = 128256
    
class BatchInferenceService:
    def __init__(self, rank: int):
        self.value_head = ValueHead(hidden_size).to(f"cuda:{rank}").eval()
        self.weights_lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = sgl.Engine(model_path=model_name, enable_return_hidden_states=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False).to(dtype=torch.bfloat16, device=f"cuda:{rank}")

        # run every 8 requests
        self.batch_size = 8
        self.rank = rank
        self.max_wait_ms = 240000

        self.per_gpu_queue = queue.Queue()
    
        self.load_lm_head()

    def load_lm_head(self):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(f"cuda:{self.rank}")
        lm_head_weight = model.lm_head.weight.detach().clone()
        self.lm_head.weight.data.copy_(lm_head_weight)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    def sync_weights(self):
        if os.path.exists(value_path) and os.path.exists(policy_path):
            with self.weights_lock:
                state_dict = torch.load(value_path, map_location=f"cuda:{self.rank}", weights_only=True)
                self.value_head.load_state_dict(state_dict)
                self.llm.update_weights_from_disk(policy_path)
            print(f"Rank {self.rank}: Loaded new weights")

    def weight_subscriber(self):
        redis = Redis(host="localhost", port=6379, db=0)
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
    def run_batch(self, input_ids, topk):
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": 1,
        }

        with self.weights_lock:
            outputs = self.llm.generate(input_ids=input_ids, sampling_params=sampling_params, return_hidden_states=True, return_logprob=True, top_logprobs_num=topk)
            policies = []
            last_hidden_states = []
            values = []

            for output in outputs:
                meta_info = output["meta_info"]
                prefill_store = meta_info["hidden_states"][0]
                if len(prefill_store) == 0:
                    raise ValueError("Prefill store is empty")

                last_hidden_state = torch.tensor(prefill_store[-1], dtype=torch.bfloat16).cuda() # [hidden_size]
                last_hidden_states.append(last_hidden_state)

            last_hidden_states = torch.stack(last_hidden_states)
            values : torch.Tensor = self.value_head(last_hidden_states.float()).squeeze(-1) # [batch_size]
            logits = self.lm_head(last_hidden_states)

        values = values.cpu().tolist()
        top_logits, top_indices = torch.topk(logits, topk, dim=-1)

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

            try:
                policy, value = self.run_batch(batch_input_ids, topk)
                for fut, pol, val in zip(futures, policy, value):
                    fut.set_result((pol, val))
            except Exception as e:
                for fut in futures:
                    fut.set_exception(e)
            

class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self, batch_inference_service: BatchInferenceService):
        self.batch_inference_service = batch_inference_service
        self.graders = Graders()

    def infer(self, request : InferenceRequest, context):
        fut = Future()
        state = list(request.state)
        self.batch_inference_service.per_gpu_queue.put((state, fut))
        policies, values = fut.result(timeout=3 * self.batch_inference_service.max_wait_ms / 1000.0)
        return inference_pb2.InferenceResponse(policy=policies, value=values)
    
    def grader(self, request : GraderRequest, context):
        state = request.state
        prompt_id = request.prompt_id
        string_state = self.batch_inference_service.tokenizer.decode(state)
        # parse answer in <answer>...</answer>
        reward = self.graders.maths_grader(string_state, prompt_id)
        return inference_pb2.GraderResponse(reward=reward)


def test(rank: int):
    worker = BatchInferenceService(rank)
    messages = [{"role": "user", "content": "Complete the sentence with just one word: The capital of France is: "}]
    test_state = worker.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    test_state = worker.tokenizer.encode(test_state)

    test_policy, test_value = worker.run_batch([test_state], topk)
    print(f"Rank {rank}: {test_policy}")
    print(f"Rank {rank}: {test_value}")
    max_token = max(test_policy[0], key=test_policy[0].get)
    print(f"Rank {rank}: {max_token}")
    decoded = worker.tokenizer.decode([max_token])
    print(f"Rank {rank}: {decoded}")



def serve():
    rank = int(os.environ.get("RANK", 0))
    port = int(os.environ.get("PORT", 50051 + rank))

    worker = BatchInferenceService(rank)
    threading.Thread(target=worker.batch_worker, daemon=True).start()
    threading.Thread(target=worker.weight_subscriber, daemon=True).start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(worker), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f'Server started on port {port}')
    server.wait_for_termination()

if __name__ == "__main__":
    serve()