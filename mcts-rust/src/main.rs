use anyhow::Result;
use mcts_rust::mcts::mcts;

#[tokio::main]
async fn main() -> Result<()> {
    let state = vec![1, 2, 3, 4];
    let result = mcts(state).await?;
    println!("Result: {}", result);
    Ok(())
}
