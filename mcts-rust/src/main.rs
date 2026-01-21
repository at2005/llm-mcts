pub mod mcts;
use mcts::Tree;
pub mod gpu_runner;
pub mod grpc;

fn main() {
    let tree = Tree::new();
    println!("Hello, world!");
}
