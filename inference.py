import grpc
from concurrent import futures
import inference_pb2_grpc
import inference_pb2
import random

class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def infer(self, request, context):
        prior = {1: random.random(), 2: random.random(), 3: random.random()}
        value = random.random()
        return inference_pb2.InferenceResponse(prior=prior, value=value)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('Server started on port 50051')
    server.wait_for_termination()