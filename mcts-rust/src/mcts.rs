use crate::inference::inference_client::InferenceClient;
use anyhow::Result;
use std::cell::UnsafeCell;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use tch::{Device, Kind, Tensor};
use tonic::transport::Channel;

type NodeId = usize;
use crate::grpc::{get_client, policy_value_head};
const C_PUCT: f32 = 1.0;
const VIRTUAL_LOSS: u32 = 1;
const NUM_NODES_TO_EXPAND: usize = 100;
const MAX_ITERATIONS: usize = 1000;
const EOS_ACTION: u32 = 1;

pub fn to_vec_f32(t: &Tensor) -> Vec<f32> {
    let t = t.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
    let n = t.numel();
    let mut v = vec![0f32; n];
    t.copy_data(&mut v, n);
    v
}

pub fn to_vec_i64(t: &Tensor) -> Vec<i64> {
    let t = t.to_device(Device::Cpu).to_kind(Kind::Int64).contiguous();
    let n = t.numel();
    let mut v = vec![0i64; n];
    t.copy_data(&mut v, n);
    v
}

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

pub struct Node {
    visits: AtomicU32,
    value: AtomicF32,
    parent: Option<NodeId>,
    action: Option<u32>,
    state: Vec<u64>,
    prior: f32,
    children: Vec<NodeId>,
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
            children: vec![],
        }
    }
}

pub struct Tree {
    nodes: UnsafeCell<Vec<Node>>,
    size: AtomicUsize,
    mutex: Mutex<()>,
    inference_client: InferenceClient<Channel>,
}

impl Tree {
    pub async fn new() -> Result<Self> {
        let inference_client = get_client().await?;
        Ok(Self {
            nodes: UnsafeCell::new(vec![]),
            size: AtomicUsize::new(0),
            mutex: Mutex::new(()),
            inference_client: inference_client,
        })
    }

    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id: usize = self.size.fetch_add(1, Ordering::Relaxed);
        let nodes = unsafe { &mut *self.nodes.get() };

        if let Some(parent) = node.parent {
            nodes[parent].children.push(id);
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

impl Tree {
    pub fn puct(&self, node: &Node, child: &Node) -> f32 {
        let visits = node.visits.load(Ordering::Relaxed);
        let child_visits = child.visits.load(Ordering::Relaxed);
        let value = child.value.load(Ordering::Relaxed);

        let q = match visits {
            0 => 0.0,
            _ => value / visits as f32,
        };

        let exploration_term = (visits as f32).sqrt() / (1.0 + child_visits as f32);
        let puct_term = q + C_PUCT * exploration_term * child.prior;
        puct_term
    }

    pub fn select(&self, node: &Node) -> NodeId {
        let puct_terms = node
            .children
            .iter()
            .map(|child| (self.puct(node, self.get_node(*child)), *child));

        let (_score, best_child_id) = puct_terms
            .max_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN"))
            .expect("No children");
        best_child_id
    }

    pub fn backprop(&mut self, node: NodeId, reward: f32) {
        let node = self.get_node_mut(node);
        node.value.fetch_add(reward, Ordering::Relaxed);
        node.visits.fetch_add(1 - VIRTUAL_LOSS, Ordering::Relaxed);
        if let Some(parent) = node.parent {
            self.backprop(parent, reward);
        }
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

    pub async fn expand(&mut self, node: NodeId) -> Result<(BTreeMap<u32, f32>, f32)> {
        let current_node = self.get_node(node);
        let current_state = current_node.state.clone();

        let (prior, value) = policy_value_head(
            &mut self.inference_client,
            &current_state,
            NUM_NODES_TO_EXPAND,
        )
        .await?;

        let _guard = self.mutex.lock().unwrap();
        let nodes = unsafe { &mut *self.nodes.get() };

        let indices = prior.keys().collect::<Vec<_>>();

        for &action in indices.iter() {
            let new_state = self._build_new_state(node, *action as u64);
            let new_node = Node::new(Some(node), Some(*action as u32), new_state, prior[action]);
            let id: usize = self.size.fetch_add(1, Ordering::Relaxed);
            nodes[node].children.push(id);
            nodes.push(new_node);
        }

        Ok((prior, value))
    }
}

pub async fn mcts(state: Vec<u64>) -> Result<NodeId> {
    let mut tree = Tree::new().await?;
    let root_node = Node::new(None, None, state.clone(), 1.0);
    let root_id = tree.add_node(root_node);

    let mut iterations = 0;

    while iterations < MAX_ITERATIONS {
        let mut node = root_id;
        while !tree.is_leaf(node)? {
            node = tree.select(tree.get_node(node));
            if tree.get_node(node).action == Some(EOS_ACTION) {
                break;
            }
            continue;
        }
        let (_prior, value) = tree.expand(node).await?;
        tree.backprop(node, value);
        iterations += 1;
    }

    let root = tree.get_root();
    let most_visited = root
        .children
        .iter()
        .max_by(|a, b| {
            let a_visits = tree.get_node(**a).visits.load(Ordering::Relaxed);
            let b_visits = tree.get_node(**b).visits.load(Ordering::Relaxed);
            a_visits.partial_cmp(&b_visits).unwrap()
        })
        .expect("No children")
        .clone();

    Ok(most_visited)
}
