use anyhow::Result;
use mcts_rust::mcts::spawn_mcts_workers;

#[tokio::main]
async fn main() -> Result<()> {
    let num_workers = std::env::var("BATCH_WORKER_SIZE")
        .unwrap_or("100".to_string())
        .parse::<usize>()
        .unwrap();
    let handles = (0..num_workers)
        .map(|i| {
            let worker_pool_id = i as u32;
            spawn_mcts_workers(worker_pool_id, 1000)
        })
        .collect::<Vec<_>>();
    futures::future::join_all(handles).await;
    Ok(())
}
