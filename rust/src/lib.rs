use pyo3::{prelude::*, types::PyModule};
use pyo3::wrap_pyfunction;

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;


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

    // 3) Turn owned handle (Py<PyAny>) into borrowed &PyAny for this GIL scope
    let board_ref = board.bind(py);

    // get list of legal actions at the root
    let raw_actions:Vec<Py<PyAny>> = py_list_actions
        .call1((board_ref, player_index))?
        .extract()?;

    if raw_actions.is_empty(){
        let value: f32 = py_evaluate_board
        .call1((board_ref, player_index))?
        .extract()?;
        return Ok((value, board.clone_ref(py)));


    }
    // Stats per root action: visits and value_sum
    struct RootStat {
        action: Py<PyAny>,
        visits: u32,
        value_sum: f32,
    }

    let mut stats: Vec<RootStat> = raw_actions
        .into_iter()
        .map(|a| RootStat {
            action: a,
            visits: 0,
            value_sum: 0.0,
        })
        .collect();

    // Random number generator for simulations

    let mut rng = SmallRng::seed_from_u64(12345);

    // MCTS iterations
    //sample actions at root
    //apply action
    //evaluate resulting board
    //update stats
     for _ in 0..iterations {
        // choose random action index
        let idx = rng.random_range(0..stats.len());
        let entry = &mut stats[idx];

        // call apply_action(root_board, action) in Python
        let action_ref = entry.action.bind(py);
        let new_board_obj = py_apply_action
            .call1((&board_ref, &action_ref))?;
        let new_board: Py<PyAny> = new_board_obj.into();

        // evaluate_board(new_board, root_player)
        let new_board_ref = new_board.bind(py);
        let value: f32 = py_evaluate_board
            .call1((&new_board_ref, player_index))?
            .extract()?;

        entry.visits += 1;
        entry.value_sum += value;
    }
    
// choose best action on highest MEAN

    let mut best_idx = 0usize;
    let mut best_mean = f32::NEG_INFINITY;

    for(i,s) in stats.iter().enumerate(){
        if s.visits == 0 {
            continue;
        }

        let mean = s.value_sum / (s.visits as f32);
        if mean > best_mean {
            best_mean = mean;
            best_idx = i;
    }

}

 //if iter ==0 fallback
    if stats[best_idx].visits == 0 {
        let value:f32 = py_evaluate_board
        .call1((board_ref, player_index))?
        .extract()?;
        return Ok((value, stats[0].action.clone_ref(py)));
    }

    let best_action = stats[best_idx].action.clone_ref(py);
    let best_value = best_mean;

    Ok((best_value, best_action))
}

#[pymodule]
fn rust(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_mcts_python_rules, m)?)?;
    Ok(())
}