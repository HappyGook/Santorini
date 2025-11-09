use pyo3::wrap_pyfunction;
use pyo3::{prelude::*, types::PyModule};
use rand::{SeedableRng, rngs::SmallRng};
pub mod mcts_tree;
use mcts_tree::{Node, Tree};

// cd Santorini/rust
// cargo build --release
// maturin build --release
// python -m pip install --force-reinstall target\wheels\rust-0.1.0-cp312-cp312-win_amd64.whl

const MAX_DEPTH: usize = 8; // limit search depth for safety

fn next_player(p: u8) -> u8 {
    1 - p // assumes 2 players: 0 <-> 1
}

fn backpropagate(tree: &mut Tree, path: &[usize], value: f32) {
    for &idx in path {
        let node = &mut tree.nodes[idx];
        node.visits += 1;
        node.value_sum += value; // value is from root player's perspective
    }
}
#[pyfunction]

fn run_mcts_python_rules(
    py: Python,
    board: Py<PyAny>,
    player_index: u8,
    iterations: u32,
) -> PyResult<(f32, Py<PyAny>)> {
    //  Import helper module via PyO3
    let helpers = py.import("ai.rust_helpers")?;

    //  Get references to Python functions
    let py_list_actions = helpers.getattr("list_actions")?;
    let py_evaluate_board = helpers.getattr("evaluate_board")?;
    let py_apply_action = helpers.getattr("apply_action")?;
    let py_is_terminal = helpers.getattr("is_terminal")?;

    // Build tree with root
    let mut tree = Tree::new(board.clone_ref(py), player_index);
    let root_idx = 0usize;

    let mut _rng = SmallRng::seed_from_u64(12345);

    for _iter in 0..iterations {
        //Optional debug
        // if iter % 50 == 0 {
            
        //     //println!("[Rust PUCT] iter {iter}/{iterations}");
        // }

        // 1) Selection: walk down the tree using PUCT
        let mut path: Vec<usize> = Vec::with_capacity(MAX_DEPTH + 1);
        let mut node_idx = root_idx;
        path.push(node_idx);

        loop {
            let node = &tree.nodes[node_idx];

            if node.children.is_empty() || path.len() >= MAX_DEPTH {
                break; // reached a leaf or depth limit
            }

            if let Some(child_idx) = tree.select_child(node_idx) {
                node_idx = child_idx;
                path.push(node_idx);
            } else {
                break;
            }
        }

        // 2) Leaf node
        let leaf_idx = *path.last().unwrap();
        let leaf_player = tree.nodes[leaf_idx].player_to_move;
        let leaf_board = tree.nodes[leaf_idx].board.clone_ref(py);
        let leaf_board_ref = leaf_board.bind(py);

        // 3) Check terminal / generate actions
        let actions: Vec<Py<PyAny>> = py_list_actions
            .call1((leaf_board_ref.clone(), leaf_player))?
            .extract()?;

        let leaf_value: f32;

        let is_term: bool = py_is_terminal
            .call1((leaf_board_ref.clone(), player_index))?
            .extract()?;
        if is_term {
            leaf_value = 1.0; // treat as very good for root player
        } else if actions.is_empty() {
            // No actions (stuck position) -> just evaluate board
            leaf_value = py_evaluate_board
                .call1((leaf_board_ref.clone(), player_index))?
                .extract()?;
        } else {
            // a) Evaluate current leaf board once
            leaf_value = py_evaluate_board
                .call1((leaf_board_ref.clone(), player_index))?
                .extract()?;

            // b) Expand children on first visit
            if tree.nodes[leaf_idx].children.is_empty() {
                let prior_per_action = 1.0f32 / actions.len() as f32;
                for act in &actions {
                    let child_board: Py<PyAny> = py_apply_action
                        .call1((leaf_board_ref.clone(), act.bind(py)))?
                        .extract()?;

                    let mut child =
                        Node::new(child_board, next_player(leaf_player), prior_per_action);
                    child.action_from_parent = Some(act.clone_ref(py));
                    tree.add_child(leaf_idx, child);
                }
            }
        }
        // 4) Backpropagate value up the path
        backpropagate(&mut tree, &path, leaf_value);
    }

    // 5) Choose best child at root (highest visit count)
    let root = &tree.nodes[root_idx];
    if root.children.is_empty() {
        // No moves: evaluate root board and return "no action" (None)
        let root_ref = board.bind(py);
        let value: f32 = py_evaluate_board
            .call1((root_ref, player_index))?
            .extract()?;

        // Python None as the action, like original mcts_search does (None action)
        let none_action: Py<PyAny> = py.None().into();
        return Ok((value, none_action));
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
    let best_action = best_child
        .action_from_parent
        .as_ref()
        .unwrap()
        .clone_ref(py);

    Ok((best_value, best_action))
}

#[pymodule]
fn rust(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_mcts_python_rules, m)?)?;
    Ok(())
}
