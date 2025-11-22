pub mod board;
pub mod heuristics;
pub mod mcts_tree;


use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use crate::board::{Board, Action};
use crate::heuristics::evaluate_mcts;
use mcts_tree::{Node, Tree};
use pyo3::types::PyTuple;
const MAX_DEPTH: usize = 8;



pub fn next_player(p: u8, num_players: u8) -> u8 {
    (p + 1) % num_players
}

fn backpropagate(tree: &mut Tree, path: &[usize], value: f32) {
    for &idx in path {
        let node = &mut tree.nodes[idx];
        node.visits += 1;
        node.value_sum += value;
    }
}

fn action_to_py(py: Python<'_>, act: Action) -> Py<PyAny> {
    // (move_x, move_y)
    let move_tuple = PyTuple::new(py, &[act.move_to.x as i32, act.move_to.y as i32])
        .expect("failed to create move tuple");

    // (build_x, build_y)
    let build_tuple = PyTuple::new(py, &[act.build_at.x as i32, act.build_at.y as i32])
        .expect("failed to create build tuple");

    // (worker_index, (move_x, move_y), (build_x, build_y))
    let action_tuple = PyTuple::new(
        py,
        &[
            act.worker_index as i32,
            move_tuple.into(),
            build_tuple.into(),
        ],
    )
    .expect("failed to create action tuple");

    // Bound<PyTuple> -> Py<PyTuple> -> Py<PyAny>
    let action_pytuple: Py<PyTuple> = action_tuple.into();
    let py_action: Py<PyAny> = action_pytuple.into();
    py_action
}

fn board_from_python(
    py: Python<'_>,
    board: &Py<PyAny>,
    num_players: u8,
    current_player: u8,
) -> PyResult<Board> {
    let bound = board.bind(py);
  
    Board::from_python(py, bound.clone(), num_players, current_player)
}

#[pyfunction]
fn run_mcts_python_rules(
    py: Python<'_>,
    board: Py<PyAny>,
    player_index: u8,
    iterations: u32,
    num_players: u8,
) -> PyResult<(f32, Py<PyAny>)> {
    // Convert root board to Rust and create tree
    let rust_root_board = board_from_python(py, &board, num_players, player_index)?;
    let mut tree = Tree::new(rust_root_board, player_index);
    let root_idx: usize = 0;

    // Root expansion
    let root_actions: Vec<Action> =
        tree.nodes[root_idx].board.legal_actions_for_current_player();

    if !root_actions.is_empty() {
        let prior = 1.0f32 / root_actions.len() as f32;
        for act in &root_actions {
            let child_board = tree.nodes[root_idx].board.apply_action(*act);
            let mut child = Node::new(
                child_board,
                next_player(player_index, num_players),
                prior,
            );
            child.action_from_parent = Some(*act);
            tree.add_child(root_idx, child);
        }
    }

    // MCTS iterations
    for _ in 0..iterations {
        // Selection
        let mut path: Vec<usize> = Vec::with_capacity(MAX_DEPTH + 1);
        let mut node_idx = root_idx;
        path.push(node_idx);

        loop {
            let node = &tree.nodes[node_idx];

            if path.len() >= MAX_DEPTH {
                break;
            }
            if node_idx != root_idx && node.children.is_empty() {
                break;
            }

            if let Some(child_idx) = tree.select_child(node_idx) {
                node_idx = child_idx;
                path.push(node_idx);
            } else {
                break;
            }
        }

        // Leaf node
        let leaf_idx = *path.last().unwrap();
        let leaf_player = tree.nodes[leaf_idx].player_to_move;

        // Actions at leaf
        let actions: Vec<Action> =
            tree.nodes[leaf_idx].board.legal_actions_for_current_player();

        // Evaluate leaf
        let leaf_value: f32 = evaluate_mcts(&tree.nodes[leaf_idx].board, player_index);

        // Expand leaf if not terminal
        if !actions.is_empty() && tree.nodes[leaf_idx].children.is_empty() {
            let prior_per_action = 1.0f32 / actions.len() as f32;

            for act in &actions {
                let child_board = tree.nodes[leaf_idx].board.apply_action(*act);
                let mut child = Node::new(
                    child_board,
                    next_player(leaf_player, num_players),
                    prior_per_action,
                );
                child.action_from_parent = Some(*act);
                tree.add_child(leaf_idx, child);
            }
        }

        // Backpropagate
        backpropagate(&mut tree, &path, leaf_value);
    }

    // Choose best child at root
    let root = &tree.nodes[root_idx];
    if root.children.is_empty() {
        let value = evaluate_mcts(&tree.nodes[root_idx].board, player_index);
        return Ok((value, py.None()));
    }

    let mut best_child_idx = root.children[0];
    let mut best_visits = tree.nodes[best_child_idx].visits;

    for &child_idx in &root.children {
        let v = tree.nodes[child_idx].visits;
        if v > best_visits {
            best_visits = v;
            best_child_idx = child_idx;
        }
    }

    let best_child = &tree.nodes[best_child_idx];
    let best_value = best_child.q_value();
    let act = best_child
        .action_from_parent
        .expect("best child must have an action");

    // Convert Rust Action -> Python tuple: (worker_index, (move_x, move_y), (build_x, build_y))
    
let py_action = action_to_py(py, act);
Ok((best_value, py_action))
    // Convert Rust Action -> Python tuple
   
}

#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_mcts_python_rules, m)?)?;
    Ok(())
}