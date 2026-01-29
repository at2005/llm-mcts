import grpc
from concurrent import futures
import inference_pb2_grpc
import inference_pb2
from inference_pb2 import InferenceRequest, GraderRequest
import torch
import threading
import sglang as sgl
from transformers import AutoTokenizer
from model import ValueHead, model_name
from concurrent.futures import Future
from graders import Graders
import time
import queue
import os
import re
import numpy as np
from typing import List, Tuple

topk = 4
hidden_size = 4096

class BatchInferenceService:
    def __init__(self, rank: int):
        self.value_head = ValueHead(hidden_size).to(f"cuda:{rank}")
        self.value_head_sync_lock = threading.Lock()
        self.value_head_path = "value_head.pth"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = sgl.Engine(model=model_name)
        # run every 8 requests
        self.batch_size = 8
        self.rank = rank
        self.max_wait_ms = 240000

        self.per_gpu_queue = queue.Queue()
    

    def sync_weights(self):
        with self.value_head_sync_lock:
            self.value_head.load_state_dict(torch.load(self.value_head_path, map_location=f"cuda:{self.rank}"))
            self.llm.load_weights(self.llm_path)
    
    def renormalize_policies(self, policies: List[List[Tuple[int, float]]]) -> List[List[Tuple[int, float]]]:
        pass

    @torch.inference_mode()
    def run_batch(self, input_ids, topk):
        sampling_params = {
            "temperature": 0.0,
            "max_tokens": 1,
        }

        outputs = self.llm.generate(input_ids, sampling_params=sampling_params, return_hidden_states=True, return_logprob=True, top_logprobs_num=topk)
        policies = []
        last_hidden_states = []
        values = []

        for output in outputs:
            meta_info = output["meta_info"]
            prefill_store = meta_info["hidden_states"][0]
            if len(prefill_store) == 0:
                raise ValueError("Prefill store is empty")
            
            last_hidden_state = torch.tensor(prefill_store[-1], dtype=torch.float32).cuda() # [hidden_size]
            last_hidden_states.append(last_hidden_state)

            top_logprobs = output["output_token_logprobs"][0][-1]

            if isinstance(top_logprobs, dict):
                policy = [(int(k), np.exp(v)) for k, v in top_logprobs.items()]
                policies.append(policy)
            else:
                raise ValueError("Top logprobs is not a dict")
            
        with self.value_head_sync_lock:
            values : torch.Tensor = self.value_head(torch.stack(last_hidden_states)).squeeze(-1) # [batch_size]
        values = values.cpu().tolist()
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
        self.batch_inference_service.per_gpu_queue.put((request.state, fut))
        policies, values = fut.result(timeout=3 * self.batch_inference_service.max_wait_ms / 1000.0)
        return inference_pb2.InferenceResponse(policy=policies, value=values)
    
    def grader(self, request : GraderRequest, context):
        state = request.state
        prompt_id = request.prompt_id
        string_state = self.batch_inference_service.tokenizer.decode(state)
        # parse answer in <answer>...</answer>
        reward = self.graders.maths_grader(string_state, prompt_id)
        return inference_pb2.GraderResponse(reward=reward)

def serve():
    rank = int(os.environ.get("RANK", 0))
    port = int(os.environ.get("PORT", 50050 + rank))

    worker = BatchInferenceService(rank)
    threading.Thread(target=worker.batch_worker, daemon=True).start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(worker), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f'Server started on port {port}')
    server.wait_for_termination()

if __name__ == "__main__":
    serve()