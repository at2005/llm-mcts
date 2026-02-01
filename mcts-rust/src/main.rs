use anyhow::Result;
use mcts_rust::config::load_config;
use mcts_rust::mcts::spawn_mcts_workers;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let num_workers = std::env::var("BATCH_WORKER_SIZE")
        .unwrap_or("1".to_string())
        .parse::<usize>()
        .unwrap();

    info!("Starting {} MCTS worker pools", num_workers);

    info!("Loading config");
    let config = load_config()?;
    info!("Config loaded");

    let handles = (0..num_workers)
        .map(|i| {
            let worker_pool_id = i as u32;
            info!("Spawning MCTS worker pool {}", worker_pool_id);
            tokio::spawn(spawn_mcts_workers(worker_pool_id, 1000, config.clone()))
        })
        .collect::<Vec<_>>();
    let results = futures::future::join_all(handles).await;
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(Ok(())) => info!("Worker {} completed successfully", i),
            Ok(Err(e)) => error!("Worker {} returned error: {:?}", i, e),
            Err(e) => error!("Worker {} panicked: {:?}", i, e),
        }
    }
    info!("All MCTS worker pools completed");
    Ok(())
}
