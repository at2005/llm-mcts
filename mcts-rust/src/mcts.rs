use crate::inference::inference_client::InferenceClient;
use anyhow::Result;
use std::cell::UnsafeCell;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use tonic::transport::Channel;

type NodeId = usize;
use crate::grpc::{get_client, policy_value_head};
pub const ROOT_NODE_ID: NodeId = 0;
const C_PUCT: f32 = 1.0;
const VIRTUAL_LOSS: u32 = 1;
const NUM_NODES_TO_EXPAND: usize = 100;
const MAX_ITERATIONS: usize = 1000;
const EOS_ACTION: u32 = 100;
const MAX_NODES: usize = 1000000;

#[derive(Debug)]
pub struct AtomicF32 {
    bits: AtomicU32,
}

impl AtomicF32 {
    pub fn new(v: f32) -> Self {
        Self {
            bits: AtomicU32::new(v.to_bits()),
        }
    }

    pub fn load(&self, ordering: Ordering) -> f32 {
        f32::from_bits(self.bits.load(ordering))
    }

    pub fn store(&self, value: f32, ordering: Ordering) {
        self.bits.store(value.to_bits(), ordering);
    }

    pub fn fetch_add(&self, value: f32, ordering: Ordering) -> f32 {
        let current = self.load(ordering);
        let new = current + value;
        self.store(new, ordering);
        new
    }
}

#[derive(Debug)]
pub struct Node {
    pub visits: AtomicU32,
    pub value: AtomicF32,
    pub parent: Option<NodeId>,
    pub action: Option<u32>,
    pub state: Vec<u64>,
    pub prior: f32,
    // action -> node id
    pub children: BTreeMap<u32, NodeId>,
    // action -> prior
    pub child_priors: BTreeMap<u32, f32>,
}

impl Node {
    pub fn new(parent: Option<NodeId>, action: Option<u32>, state: Vec<u64>, prior: f32) -> Self {
        Self {
            visits: AtomicU32::new(0),
            value: AtomicF32::new(0.0),
            parent: parent,
            action: action,
            state: state,
            prior: prior,
            children: BTreeMap::new(),
            child_priors: BTreeMap::new(),
        }
    }
}

pub struct Tree {
    pub nodes: UnsafeCell<Vec<Node>>,
    pub size: AtomicUsize,
    pub mutex: Mutex<()>,
    pub inference_client: InferenceClient<Channel>,
}

unsafe impl Sync for Tree {}

impl Tree {
    pub async fn new() -> Result<Self> {
        let inference_client = get_client().await?;
        let nodes = Vec::with_capacity(MAX_NODES);
        Ok(Self {
            nodes: UnsafeCell::new(nodes),
            size: AtomicUsize::new(0),
            mutex: Mutex::new(()),
            inference_client: inference_client,
        })
    }

    pub fn add_node(&mut self, node: Node) -> NodeId {
        let _guard: std::sync::MutexGuard<'_, ()> = self.mutex.lock().unwrap();
        let id: usize = self.size.fetch_add(1, Ordering::Relaxed);
        let nodes = unsafe { &mut *self.nodes.get() };

        if let Some(parent) = node.parent {
            nodes[parent]
                .children
                .insert(node.action.expect("Action is None"), id);
        }
        nodes.push(node);
        id
    }

    pub fn get_node(&self, id: NodeId) -> &Node {
        let nodes = unsafe { &*self.nodes.get() };
        &nodes[id]
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> &mut Node {
        let nodes = unsafe { &mut *self.nodes.get() };
        let node = &mut nodes[id];
        node
    }

    pub fn get_root(&self) -> &Node {
        let nodes = unsafe { &*self.nodes.get() };
        &nodes[0]
    }
}

pub fn puct(visits: u32, value: f32, child_visits: u32, prior: f32) -> f32 {
    let q = match child_visits {
        0 => 0.0,
        _ => value / child_visits as f32,
    };

    let exploration_term = (visits as f32).sqrt() / (1.0 + child_visits as f32);
    let puct_term = q + C_PUCT * exploration_term * prior;
    puct_term
}

impl Tree {
    pub fn select(&self, node: &Node, priors: &BTreeMap<u32, f32>) -> u32 {
        let children = node
            .children
            .iter()
            .map(|(action, id)| (*action, self.get_node(*id)))
            .collect::<BTreeMap<_, _>>();
        let puct_terms = priors
            .iter()
            .map(|(action, prior)| {
                let parent_visits = node.visits.load(Ordering::Relaxed);
                let puct_term = match children.get(action) {
                    None => puct(parent_visits, 0.0, 0, *prior),
                    Some(child) => puct(
                        parent_visits,
                        child.value.load(Ordering::Relaxed),
                        child.visits.load(Ordering::Relaxed),
                        *prior,
                    ),
                };
                (puct_term, *action)
            })
            .collect::<Vec<_>>();

        let best_action = puct_terms
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN"))
            .expect("Failed to find best action")
            .1;
        best_action
    }

    pub fn backprop(&mut self, node: NodeId, reward: f32) {
        let parent = self.get_node(node).parent.expect("Parent is None");
        let parent_node = self.get_node_mut(parent);
        parent_node.value.fetch_add(reward, Ordering::Relaxed);
        parent_node
            .visits
            .fetch_add(1 - VIRTUAL_LOSS, Ordering::Relaxed);
        if parent_node.parent.is_none() {
            return;
        }
        self.backprop(parent, reward);
    }

    pub fn is_leaf(&self, node: NodeId) -> Result<bool> {
        let node = self.get_node(node);
        Ok(node.children.is_empty())
    }

    pub fn _build_new_state(&self, node: NodeId, action: u64) -> Vec<u64> {
        let current_node = self.get_node(node);
        let mut new_state = current_node.state.clone();
        new_state.push(action);
        new_state
    }

    pub async fn store_policy_value(&mut self, node: NodeId) -> Result<(BTreeMap<u32, f32>, f32)> {
        let current_node = self.get_node(node);
        let (priors, value) = policy_value_head(
            &mut self.inference_client.clone(),
            &current_node.state,
            NUM_NODES_TO_EXPAND,
        )
        .await?;

        let current_node_mut = self.get_node_mut(node);
        current_node_mut.child_priors = priors.clone();
        current_node_mut.value = AtomicF32::new(value);
        Ok((priors, value))
    }

    pub async fn expand(&mut self, node: NodeId, next_action: u32) -> Result<NodeId> {
        let new_state = self._build_new_state(node, next_action as u64);
        let new_node = Node::new(Some(node), Some(next_action), new_state.clone(), 0.0);
        let new_node_id = self.add_node(new_node);
        Ok(new_node_id)
    }
}

pub async fn mcts(tree: &mut Tree) -> Result<()> {
    // populate priors and value for root node
    tree.store_policy_value(ROOT_NODE_ID).await?;

    let mut iterations = 0;

    while iterations < MAX_ITERATIONS {
        let mut node = 0;
        let mut next_action;

        // select next action
        loop {
            let node_obj = tree.get_node(node);
            // add virtual loss to node to discourage it from being selected again, for parallel mcts
            node_obj.visits.fetch_add(VIRTUAL_LOSS, Ordering::Relaxed);

            next_action = Some(tree.select(node_obj, &node_obj.child_priors));

            if next_action == Some(EOS_ACTION) {
                break;
            }

            // check if best_action child exists in node_obj.children
            let best_action_child = node_obj
                .children
                .get(&(next_action.expect("Next action is None") as u32));

            if let Some(child) = best_action_child {
                node = *child;
            } else {
                break;
            }
        }

        let new_node_id = tree
            .expand(node, next_action.expect("Next action is None"))
            .await?;

        // populate priors and value for new node
        let (_, value) = tree.store_policy_value(new_node_id).await?;

        // backpropagate value to root node
        tree.backprop(new_node_id, value);

        iterations += 1;
    }
    Ok(())
}
