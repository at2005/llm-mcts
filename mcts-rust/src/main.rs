use anyhow::Result;
use mcts_rust::mcts::{Node, Tree, mcts};
use std::sync::atomic::Ordering;

#[tokio::main]
async fn main() -> Result<()> {
    let mut tree = Tree::new().await?;
    let state = vec![1, 2, 4];
    let root_node = Node::new(None, None, state.clone(), 1.0);
    tree.add_node(root_node);

    let result = mcts(&mut tree).await?;
    println!("Result: {}", result);
    let root_node = tree.get_node(0);
    for (_, child) in root_node.children.iter() {
        let child_node = tree.get_node(*child);
        println!(
            "child node with action {:?} has visits: {:?}",
            child_node.action,
            child_node.visits.load(Ordering::Relaxed)
        );
    }
    Ok(())
}
