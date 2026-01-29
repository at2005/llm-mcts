use anyhow::Result;
use mcts_rust::mcts::{Node, Tree, get_redis_url, greedy_select, mcts};
use std::sync::{Arc, atomic::Ordering};

#[tokio::main]
async fn main() -> Result<()> {
    let episode_id = 0;
    let prompt_id = 0;
    let tree = Arc::new(Tree::new(episode_id, prompt_id).await?);
    let state = vec![1, 2, 4];
    let root_node = Node::new(None, None, state.into());
    let root_id = tree.node_alloc(root_node);

    let client = redis::Client::open(get_redis_url().as_str())?;
    let mut con = client.get_connection_manager().await?;

    let handles: Vec<_> = (0..10)
        .map(|_| {
            let tree = Arc::clone(&tree);
            tokio::spawn(async move { mcts(&tree).await })
        })
        .collect();

    futures::future::join_all(handles).await;

    let root = tree.get_node(root_id);

    let edges = &root.expansion.get().expect("Expansion is None").edges;
    for edge in edges.iter() {
        let child_node = tree.get_node(edge.child_id.load(Ordering::Relaxed));
        println!(
            "child node with action {:?} has visits: {:?}",
            child_node.action,
            child_node.visits.load(Ordering::Relaxed)
        );
    }

    greedy_select(&mut con, &tree).await?;

    Ok(())
}
