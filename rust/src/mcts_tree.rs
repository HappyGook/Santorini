use rand::{rngs::SmallRng, Rng}; // SeedableRng
use crate::board::{Board, Action};
//use crate::heuristics::evaluate_mcts;

/// Exploration constant for PUCT (√2 is a common default)
pub const C_PUCT: f32 = 1.414;

/// A single node

pub struct Node {
    pub board: Board,             // current board state
    pub prior: f32,                      // prior probability P(a|s)
    pub value_sum: f32,                  // sum of simulation values
    pub visits: u32,                     // N(s,a)
    pub player_to_move: u8,
    pub action_from_parent: Option<Action>,
    pub children: Vec<usize>,            
}

impl Node {
    pub fn new(board: Board, player_to_move: u8, prior: f32) -> Self {
        Self {
            board,
            prior,
            value_sum: 0.0,
            visits: 0,
            player_to_move,
            action_from_parent: None,
            children: Vec::new(),
        }
    }

    /// Mean value Q = W / N
    pub fn q_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

/// Whole search tree.
pub struct Tree {
    pub nodes: Vec<Node>,
}

impl Tree {
    /// Create tree with a single root node.
    pub fn new(root_board: Board, root_player: u8) -> Self {
        let root = Node::new(root_board, root_player, 1.0);
        Self { nodes: vec![root] }
    }
// add child node to parent
    pub fn add_child(&mut self, parent_idx: usize, child: Node) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(child);
        self.nodes[parent_idx].children.push(idx);
        idx
    }
    /// Choose the best child using PUCT formula.
    /// Returns the index of the chosen child.
    pub fn select_child(&self, node_idx: usize) -> Option<usize> {
        let parent = &self.nodes[node_idx];
        if parent.children.is_empty() {
            return None;
        }


        let n_parent = parent.visits.max(1);
        let sqrt_n = (n_parent as f32).sqrt();


        let mut best_score = f32::NEG_INFINITY;
        let mut best_child = None;

        for &child_idx in &parent.children {

            let child = &self.nodes[child_idx];
            let q = child.q_value();
            let u = C_PUCT * child.prior * sqrt_n / (1.0 + child.visits as f32);
            let score = q + u;

            if score > best_score {
                best_score = score;
                best_child = Some(child_idx);
            }
        }
        best_child
    }

    /// Randomly select one child if all have 0 visits (tie-break)
    pub fn select_random_child(&self, node_idx: usize, rng: &mut SmallRng) -> Option<usize> {
        let parent = &self.nodes[node_idx];
        if parent.children.is_empty() {
            return None;
        }
        let i = rng.random_range(0..parent.children.len());
        Some(parent.children[i])
    }
}

