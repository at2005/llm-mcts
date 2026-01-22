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

    let root = tree.get_root();
    let most_visited = root
        .children
        .iter()
        .max_by(|(_, a_id), (_, b_id)| {
            let a_visits = tree.get_node(**a_id).visits.load(Ordering::Relaxed);
            let b_visits = tree.get_node(**b_id).visits.load(Ordering::Relaxed);
            a_visits.partial_cmp(&b_visits).unwrap()
        })
        .expect("No children")
        .clone();

    println!("Most visited child: {:?}", most_visited.0);

    println!("Result: {:?}", result);
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
