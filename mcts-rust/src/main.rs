use anyhow::Result;
use mcts_rust::mcts::{Node, Tree, mcts};
use std::sync::atomic::Ordering;

#[tokio::main]
async fn main() -> Result<()> {
    let mut tree = Tree::new().await?;
    let state = vec![1, 2, 4];
    let root_node = Node::new(None, None, state.into());
    let root_id = tree.node_alloc(root_node);
    println!("Root id: {:?}", root_id);
    let result = mcts(&mut tree).await?;
    let root = tree.get_node(root_id);

    println!("Result: {:?}", result);
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
