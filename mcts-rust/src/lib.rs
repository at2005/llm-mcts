pub mod config;
pub mod grpc;
pub mod logger;
pub mod mcts;

// Generated protobuf code lives under this module; tonic pulls it from OUT_DIR at build time.
pub mod inference {
    tonic::include_proto!("inference");
}
