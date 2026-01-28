import grpc
from concurrent import futures
import inference_pb2_grpc
import inference_pb2
from inference_pb2 import InferenceRequest
import torch
import threading
import sglang as sgl
from model import ValueHead, model_name
from concurrent.futures import Future
import time
import queue
import os


topk = 4
hidden_size = 4096

class BatchInferenceService:
    def __init__(self, rank: int):
        self.value_head = ValueHead(hidden_size).to(f"cuda:{rank}")
        self.value_head_sync_lock = threading.Lock()
        self.value_head_path = "value_head.pth"

        self.llm = sgl.Engine(model=model_name)
        # run every 8 requests
        self.batch_size = 8
        self.rank = rank
        self.max_wait_ms = 240000

        self.per_gpu_queue = queue.Queue()
    

    def sync_value_head(self):
        with self.value_head_sync_lock:
            self.value_head.load_state_dict(torch.load(self.value_head_path, map_location=f"cuda:{self.rank}"))
    
    @torch.inference_mode()
    def run_batch(self, input_ids, topk):
        sampling_params = {
            "temperature": 0.0,
            "max_tokens": 1,
        }

        outputs = self.llm.generate(input_ids, sampling_params=sampling_params, return_hidden_states=True, return_logprob=True, top_logprobs_num=topk)
        policies = []
        values = []
        for output in outputs:
            meta_info = output["meta_info"]
            prefill_store = meta_info["hidden_states"][0]
            if len(prefill_store) == 0:
                raise ValueError("Prefill store is empty")
            
            last_hidden_state = torch.tensor(prefill_store[-1], dtype=torch.float32).cuda()
            value = self.value_head(last_hidden_state).item()
            values.append(value)

            top_logprobs = output["output_token_logprobs"][0][-1]
            if isinstance(top_logprobs, dict):
                policy = sorted(top_logprobs.items(), key=lambda x: x[1], reverse=True)
                policies.append(policy)
            else:
                raise ValueError("Top logprobs is not a dict")
            
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

    def infer(self, request : InferenceRequest, context):
        fut = Future()
        self.batch_inference_service.per_gpu_queue.put((request.state, fut))
        policies, values = fut.result(timeout=3 * self.batch_inference_service.max_wait_ms / 1000.0)
        return inference_pb2.InferenceResponse(policy=policies, value=values)

def serve():
    rank = int(os.environ.get("RANK", 0))
    port = int(os.environ.get("PORT", 50050 + rank))

    worker = BatchInferenceService(rank)
    threading.Thread(target=worker.batch_worker, daemon=True).start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(worker), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f'Server started on port {port}')
    server.wait_for_termination()