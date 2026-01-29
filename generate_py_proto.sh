#!/bin/bash
python -m grpc_tools.protoc \
  -I mcts-rust/proto \
  --python_out=. \
  --grpc_python_out=. \
  mcts-rust/proto/inference.proto
