use anyhow::Result;
use mcts_rust::mcts::{Node, Tree, mcts};
use std::sync::{Arc, atomic::Ordering};

#[tokio::main]
async fn main() -> Result<()> {
    let tree = Arc::new(Tree::new().await?);
    let state = vec![1, 2, 4];
    let root_node = Node::new(None, None, state.into());
    let root_id = tree.node_alloc(root_node);

    let handles: Vec<_> = (0..300)
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
    Ok(())
}
