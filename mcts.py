from asyncio import taskgroups
from config import c_puct, expansion_threshold, num_nodes_to_expand, virtual_loss, policy_strategy, max_iters, num_threads
from gpu_runner import value_fn, policy_fn
import numpy as np
import torch
import torch.nn.functional as F
import random
import asyncio

class Node:
    def __init__(self, model, state, prior, parent=None, action=None):
        self.model = model
        self.visits = 0
        self.value = 0
        self.parent : Node = parent
        self.action = action
        self.state = state
        self.prior = prior
        self.children : set[Node] = {}

    def puct(self, child : 'Node'):
        q = child.value / child.visits if child.visits > 0 else 0
        exploration_term = np.sqrt(self.visits) / (1 + child.visits)
        puct_term = q + c_puct * exploration_term * child.prior
        return puct_term
    
    def select(self):
        best_child = None
        best_puct = -float('inf')
        for child in self.children:
            puct_term = self.puct(child)
            if puct_term > best_puct:
                best_puct = puct_term
                best_child = child
        return best_child
    
    def update_state(self, previous_state : torch.Tensor, action : int):
        # previous_state shape is (b, seq)
        new_state = torch.cat([previous_state, torch.tensor([action], dtype=torch.long, device=previous_state.device)], dim=-1)
        return new_state
    
    def _sample_top_actions(self, prior : torch.Tensor):
        top_actions = torch.topk(prior, k=num_nodes_to_expand).indices.cpu().numpy()
        return top_actions

    def expand(self):
        logits : torch.Tensor = policy_fn(self.model, self.state) 
        prior = F.softmax(logits, dim=0)
        top_actions = self._sample_top_actions(prior)

        for action in top_actions:
            new_node = Node(self.model, self.state, prior[action].item(), self, action)
            self.children.add(new_node)
        
    def backprop(self, reward):
        self.value += reward
        self.visits += (1 - virtual_loss)
        if self.parent:
            self.parent.backprop(reward)
        
    def is_leaf(self):
        return not self.children

def choose_next(root: Node):
    if policy_strategy == "softmax":
        visit_counts = torch.tensor([child.visits for child in root.children])
        prior = F.softmax(visit_counts, dim=0)
        sample_idx = torch.multinomial(prior, 1).item()
        return root.children[sample_idx]
    elif policy_strategy == "puct":
        return root.select()
    elif policy_strategy == "random":
        return random.choice(root.children)
    elif policy_strategy == "greedy":
        return max(root.children, key=lambda child: child.visits)
    else:
        raise ValueError(f"Invalid policy strategy: {policy_strategy}")


async def mcts(model, root : Node):
    node = root
    num_iters = 0

    while num_iters < max_iters:
        while True:
            if node.is_leaf():
                node.expand()
                node_value = value_fn(model, node.state)
                node.backprop(node_value)
                node = root
                break
            node = node.select()
            previous_state = node.state
            node.state = node.update_state(previous_state, node.action)
            node.visits += virtual_loss
    num_iters += 1


async def multithread_mcts(model, init_state, num_threads: int):
    root = Node(model, init_state, 1.0)
    async with asyncio.TaskGroup() as tg:
        for _ in range(num_threads):
            tg.create_task(asyncio.to_thread(mcts(model, root)))
    return root

