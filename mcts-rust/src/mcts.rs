use crate::inference::inference_client::InferenceClient;
use anyhow::Result;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use tonic::transport::Channel;

type NodeId = usize;
use crate::grpc::{get_client, policy_value_head};
pub const ROOT_NODE_ID: NodeId = 0;
const C_PUCT: f32 = 1.0;
const VIRTUAL_LOSS: u32 = 1;
const MAX_ITERATIONS: usize = 1000;
const EOS_ACTION: u32 = 100;
const MAX_NODES: usize = 1000000;
pub const NO_CHILD: usize = usize::MAX;
pub const EXPANDING_NODE: usize = usize::MAX - 1;

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
        loop {
            let current = self.load(ordering);
            let new = current + value;
            match self
                .bits
                .compare_exchange(current.to_bits(), new.to_bits(), ordering, ordering)
            {
                Ok(_) => return new,
                Err(_) => continue,
            }
        }
    }
}

#[derive(Debug)]
pub struct Edge {
    pub child_id: AtomicUsize,
    pub prior: f32,
    pub action: u32,
}

impl Edge {
    pub fn new(prior: f32, action: u32) -> Self {
        Self {
            child_id: AtomicUsize::new(NO_CHILD),
            prior,
            action,
        }
    }
}

#[derive(Debug)]
pub struct Expansion {
    pub edges: Arc<[Edge]>,
    value: f32,
}

#[derive(Debug)]
pub struct Node {
    pub visits: AtomicU32,
    pub parent: Option<NodeId>,
    pub action: Option<u32>,
    pub accumulated_value: Arc<AtomicF32>,
    pub state: Arc<[u64]>,
    pub expansion: tokio::sync::OnceCell<Expansion>,
}

impl Node {
    pub fn new(parent: Option<NodeId>, action: Option<u32>, state: Arc<[u64]>) -> Self {
        Self {
            visits: AtomicU32::new(0),
            parent,
            action,
            state,
            expansion: tokio::sync::OnceCell::new(),
            accumulated_value: Arc::new(AtomicF32::new(0.0)),
        }
    }

    pub async fn ensure_expansion(&self, tree: &Tree) -> Result<&Expansion> {
        self.expansion
            .get_or_try_init(|| async move {
                let (priors, value) =
                    policy_value_head(&mut tree.inference_client.clone(), &self.state).await?;
                let edges: Vec<Edge> = priors
                    .iter()
                    .map(|(action, prior)| Edge::new(*prior, *action))
                    .collect();

                self.accumulated_value.store(value, Ordering::Relaxed);

                Ok(Expansion {
                    edges: Arc::from(edges),
                    value,
                })
            })
            .await
    }
}

pub struct Tree {
    pub nodes: Box<[OnceLock<Node>]>,
    pub inference_client: InferenceClient<Channel>,
    pub size: AtomicUsize,
}

impl Tree {
    pub async fn new() -> Result<Self> {
        let inference_client = get_client().await?;
        let mut nodes = Vec::with_capacity(MAX_NODES);
        nodes.resize_with(MAX_NODES, OnceLock::new);

        Ok(Self {
            nodes: nodes.into(),
            inference_client: inference_client,
            size: AtomicUsize::new(0),
        })
    }

    pub fn node_alloc(&self, node: Node) -> NodeId {
        let id = self.size.fetch_add(1, Ordering::Relaxed);
        self.nodes[id].set(node).expect("slot already taken");
        id
    }

    pub fn get_node(&self, id: NodeId) -> &Node {
        self.nodes[id].get().expect("node uninitialized")
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
    pub fn select(&self, node: &Node, expansion: &Expansion) -> Result<usize> {
        let node_visits = node.visits.load(Ordering::Relaxed);
        let edge_idx = expansion
            .edges
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let a_child_id = a.child_id.load(Ordering::Relaxed);
                let b_child_id = b.child_id.load(Ordering::Relaxed);

                let a_value = match a_child_id {
                    NO_CHILD => 0.0,
                    EXPANDING_NODE => 0.0,
                    _ => self
                        .get_node(a_child_id)
                        .accumulated_value
                        .load(Ordering::Relaxed),
                };

                let b_value = match b_child_id {
                    NO_CHILD => 0.0,
                    EXPANDING_NODE => 0.0,
                    _ => self
                        .get_node(b_child_id)
                        .accumulated_value
                        .load(Ordering::Relaxed),
                };

                let a_child_visits = match a_child_id {
                    NO_CHILD => 0,
                    EXPANDING_NODE => 0,
                    _ => self.get_node(a_child_id).visits.load(Ordering::Relaxed),
                };

                let b_child_visits = match b_child_id {
                    NO_CHILD => 0,
                    EXPANDING_NODE => 0,
                    _ => self.get_node(b_child_id).visits.load(Ordering::Relaxed),
                };

                let a_puct = puct(node_visits, a_value, a_child_visits, a.prior);
                let b_puct = puct(node_visits, b_value, b_child_visits, b.prior);
                a_puct.partial_cmp(&b_puct).expect("NaN")
            })
            .expect("Failed to find best action")
            .0;
        Ok(edge_idx)
    }

    pub fn backprop(&self, node: NodeId, reward: f32) -> Result<()> {
        let node_obj = self.get_node(node);

        node_obj
            .accumulated_value
            .fetch_add(reward, Ordering::Relaxed);

        node_obj
            .visits
            .fetch_add(1 - VIRTUAL_LOSS, Ordering::Relaxed);

        if let Some(parent) = node_obj.parent {
            self.backprop(parent, reward)?;
        }

        Ok(())
    }

    pub fn is_leaf(&self, node: NodeId) -> Result<bool> {
        let node = self.get_node(node);
        Ok(node.expansion.get().is_none())
    }

    pub fn build_new_state(&self, node: NodeId, action: u64) -> Arc<[u64]> {
        let current_node = self.get_node(node);
        current_node
            .state
            .iter()
            .copied()
            .chain(std::iter::once(action))
            .collect::<Vec<_>>()
            .into()
    }

    pub async fn child_alloc(&self, parent: NodeId, edge: &Edge) -> Result<NodeId> {
        loop {
            let child_id = edge.child_id.load(Ordering::Acquire);
            if child_id != NO_CHILD && child_id != EXPANDING_NODE {
                return Ok(child_id);
            }

            match edge.child_id.compare_exchange(
                NO_CHILD,
                EXPANDING_NODE,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // we have acquired the right to allocate the child
                    let new_node = Node::new(
                        Some(parent),
                        Some(edge.action),
                        self.build_new_state(parent, edge.action as u64),
                    );
                    let new_node_id = self.node_alloc(new_node);
                    edge.child_id.store(new_node_id, Ordering::Release);
                    return Ok(new_node_id);
                }
                Err(child_id) => {
                    if child_id == EXPANDING_NODE {
                        loop {
                            let child_id = edge.child_id.load(Ordering::Acquire);
                            if child_id != EXPANDING_NODE {
                                return Ok(child_id);
                            }
                            std::hint::spin_loop();
                        }
                    }
                }
            }
        }
    }

    pub async fn expand(&self, node: NodeId, edge: &Edge) -> Result<NodeId> {
        let new_node_id = self.child_alloc(node, edge).await?;
        Ok(new_node_id)
    }
}

pub async fn mcts(tree: &Tree) -> Result<()> {
    let mut iterations = 0;

    while iterations < MAX_ITERATIONS {
        let mut node = 0;

        // select next action
        loop {
            let node_obj = tree.get_node(node);
            let expansion = node_obj.ensure_expansion(tree).await?;

            // add virtual loss to node to discourage it from being selected again, for parallel mcts
            node_obj.visits.fetch_add(VIRTUAL_LOSS, Ordering::Relaxed);

            let edge_idx = tree.select(node_obj, &expansion)?;
            let edge = &expansion.edges[edge_idx];

            if edge.action == EOS_ACTION {
                break;
            }

            let child_id = edge.child_id.load(Ordering::Acquire);

            if child_id == NO_CHILD || child_id == EXPANDING_NODE {
                node = tree.expand(node, edge).await?;
                tree.get_node(node)
                    .visits
                    .fetch_add(VIRTUAL_LOSS, Ordering::Relaxed);
                break;
            }

            node = child_id;
        }

        // backpropagate value to root node
        let expansion = tree.get_node(node).ensure_expansion(tree).await?;
        tree.backprop(node, expansion.value)?;

        iterations += 1;
    }
    Ok(())
}
