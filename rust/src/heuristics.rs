use crate::board::{Board, Pos, BOARD_SIZE, MAX_LEVEL};


fn is_win_move(board: &Board, from: Pos, to: Pos) -> bool {
    let from_h = board.height_at(from);
    let to_h = board.height_at(to);
    from_h < MAX_LEVEL && to_h == MAX_LEVEL
}


/// Higher = more control over useful towers.
fn local_tower_control(board: &Board, worker_positions: &[Pos]) -> f32 {
    let mut score: f32 = 0.0;

    for &pos in worker_positions {
        let x0 = pos.x as i32;
        let y0 = pos.y as i32;

        // 8 neighbors around the worker
        for dx in -1..=1 {
            for dy in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x0 + dx;
                let ny = y0 + dy;
                if nx >= 0 && ny >= 0 && nx < BOARD_SIZE as i32 && ny < BOARD_SIZE as i32 {
                    let p = Pos { x: nx as u8, y: ny as u8 };
                    let h = board.height_at(p) as i32;
                    if h >= 1 {
                        score += h as f32;
                    }
                }
            }
        }
    }

    score
}

/// `player_id` is the internal index of the root player: 0, 1, 2, ...
pub fn evaluate_mcts(board: &Board, player_id: u8) -> f32 {
    //  1. Collect my and opponent workers 

    // Vec of (index, position)
    let mut my_workers: Vec<(usize, Pos)> = Vec::new();
    let mut opp_workers: Vec<(usize, Pos)> = Vec::new();

    for (idx, w) in board.workers.iter().enumerate() {
        if let Some(pos) = w.pos {
            if w.owner == player_id {
                my_workers.push((idx, pos));
            } else {
                opp_workers.push((idx, pos));
            }
        }
    }

  
    // If we can win in one move, very good.
    for (idx, from) in &my_workers {
        let moves = board.legal_moves_for_worker(*idx);
        for to in moves {
            if is_win_move(board, *from, to) {
                return 0.9; // "almost certain win" for root
            }
        }
    }

    // If any opponent can win in one move, very bad.
    for (idx, from) in &opp_workers {
        let moves = board.legal_moves_for_worker(*idx);
        for to in moves {
            if is_win_move(board, *from, to) {
                return -0.9; // "almost certain loss" for root
            }
        }
    }

    // --- 3. Positional heuristic ---

    // 3.1 Height advantage (same idea as Python)
    let mut my_height: f32 = 0.0;
    for (_, pos) in &my_workers {
        my_height += board.height_at(*pos) as f32;
    }

    let mut opp_height: f32 = 0.0;
    for (_, pos) in &opp_workers {
        opp_height += board.height_at(*pos) as f32;
    }

    let height_adv: f32 = my_height - opp_height;

    // 3.2 Mobility: weighted up / flat / down moves
    let mut my_mobility: f32 = 0.0;
    for (idx, from) in &my_workers {
        let src_h = board.height_at(*from) as f32;
        let moves = board.legal_moves_for_worker(*idx);
        for to in moves {
            let dst_h = board.height_at(to) as f32;
            let diff = dst_h - src_h;
            if diff == 1.0 {
                my_mobility += 2.0;
            } else if diff == 0.0 {
                my_mobility += 1.0;
            } else if diff < 0.0 {
                my_mobility += 0.5;
            }
        }
    }

    let mut opp_mobility: f32 = 0.0;
    for (idx, from) in &opp_workers {
        let src_h = board.height_at(*from) as f32;
        let moves = board.legal_moves_for_worker(*idx);
        for to in moves {
            let dst_h = board.height_at(to) as f32;
            let diff = dst_h - src_h;
            if diff == 1.0 {
                opp_mobility += 2.0;
            } else if diff == 0.0 {
                opp_mobility += 1.0;
            } else if diff < 0.0 {
                opp_mobility += 0.5;
            }
        }
    }

    let mobility: f32 = my_mobility - opp_mobility;

    // 3.3 Distance to nearest level 3 (Chebyshev), same logic as Python

    // Collect all level-3 cells
    let mut level3: Vec<Pos> = Vec::new();
    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            let p = Pos { x: x as u8, y: y as u8 };
            if board.height_at(p) == MAX_LEVEL {
                level3.push(p);
            }
        }
    }

    fn chebyshev(a: Pos, b: Pos) -> i32 {
        let dx = (a.x as i32 - b.x as i32).abs();
        let dy = (a.y as i32 - b.y as i32).abs();
        dx.max(dy)
    }

    let board_size_i = BOARD_SIZE as i32;

    let mut my_dist_sum: f32 = 0.0;
    for (_, pos) in &my_workers {
        let dist = if !level3.is_empty() {
            level3
                .iter()
                .map(|&p| chebyshev(*pos, p))
                .min()
                .unwrap_or(board_size_i)
        } else {
            board_size_i
        };
        // same trick: (BOARD_SIZE - dist) → closer gives larger positive
        my_dist_sum += (board_size_i - dist) as f32;
    }

    let mut opp_dist_sum: f32 = 0.0;
    for (_, pos) in &opp_workers {
        let dist = if !level3.is_empty() {
            level3
                .iter()
                .map(|&p| chebyshev(*pos, p))
                .min()
                .unwrap_or(board_size_i)
        } else {
            board_size_i
        };
        opp_dist_sum += (board_size_i - dist) as f32;
    }

    let dist_sum: f32 = my_dist_sum - opp_dist_sum;

    // 3.4 Tower control (using our helper)
    let my_positions: Vec<Pos> = my_workers.iter().map(|(_, p)| *p).collect();
    let opp_positions: Vec<Pos> = opp_workers.iter().map(|(_, p)| *p).collect();

    let my_tower_ctrl  = local_tower_control(board, &my_positions);
    let opp_tower_ctrl = local_tower_control(board, &opp_positions);
    let tower_ctrl: f32 = my_tower_ctrl - opp_tower_ctrl;

    
    let raw: f32 = 8.0 * height_adv + 3.0 * mobility + 6.0 * dist_sum + 4.0 * tower_ctrl;

    // Normalize to [-0.5, 0.5]
    let mut v: f32 = raw / 100.0;
    if v > 0.5 {
        v = 0.5;
    }
    if v < -0.5 {
        v = -0.5;
    }

    v
}
