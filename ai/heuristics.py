"""
This is a heuristics file: evaluation of the moves in the game.
For now I have made 3 factors:
    1. Height advantage
    2. Mobility
    3. Proximity to level 3
    4. Tactical win in one(me) /or lose (op)
"""
from typing import Dict, Any

from game.board import BOARD_SIZE
from game.rules import legal_moves, is_win_after_move, legal_builds
from game.moves import move_worker, build_block

def local_tower_control(board_state, workers):
    """
    Sum of heights of level-2/3 cells around each worker.
    Higher = more control over useful towers.
    """
    score = 0
    for w in workers:
        if w.pos is None:
            continue
        x0, y0 = w.pos
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x0 + dx, y0 + dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                    h = board_state.get_cell((nx, ny)).height
                    if h >= 1:   # count since level 1
                        score += h
    return score


def mob_score(workers, board_state=None):
    s = 0.0
    for w in workers:
        current_h = board_state.get_cell(w.pos).height
        moves = legal_moves(board_state, w.pos)
        up = flat = down = 0
        for distance in moves:
            dst_h = board_state.get_cell(distance).height
            if dst_h > current_h:
                up += 1
            elif dst_h == current_h:
                flat += 1
            else:
                down += 1
        s += 2.0 * up + 1.0 * flat + 0.5 * down
    return s

def distance_to_nearest_level3(board_state, pos):
    #Chebyshev distance
    level3_positions = [
        (x, y)
        for x in range(BOARD_SIZE)
        for y in range(BOARD_SIZE)
        if board_state.get_cell((x, y)).height == 3
    ]
    if not level3_positions:
        return BOARD_SIZE * 2

    x0, y0 = pos
    # Chebyshev distance = max(|dx|, |dy|)
    distances = [max(abs(x - x0), abs(y - y0)) for (x, y) in level3_positions]
    return min(distances)

def evaluate(board_state, player_id):
    my_workers = [w for w in board_state.workers if w.owner == player_id]
    opponent_workers = [w for w in board_state.workers if w.owner != player_id]

    # If can move onto height 3 right now -> huge bonus
    for w in my_workers:
        for dst in legal_moves(board_state, w.pos):
            if board_state.get_cell(dst).height == 3:
                return 1000.0  # immediate win sign

    # If opponent can move onto height 3 next -> big penalty
    for w in opponent_workers:
        for dst in legal_moves(board_state, w.pos):
            if board_state.get_cell(dst).height == 3:
                return -900.0

    # --- height advantage ---
    my_height = sum(board_state.get_cell(w.pos).height for w in my_workers)
    opponent_height = sum(board_state.get_cell(w.pos).height for w in opponent_workers)
    height_advantage = my_height - opponent_height

    my_mobility = mob_score(my_workers, board_state)
    opponent_mobility = mob_score(opponent_workers, board_state)
    mobility = my_mobility - opponent_mobility

    # Proximity: for each worker distance to nearest level-3 cell (reward being closer)
    my_proximity_score = 0.0
    for w in my_workers:
        dist = distance_to_nearest_level3(board_state, w.pos)
        if dist == 0:
            my_proximity_score += 10.0  # bonus for being on level 3
        else:
            my_proximity_score += 1.0 / (dist+0.5)

    opponent_proximity_score = 0.0
    for w in opponent_workers:
        dist = distance_to_nearest_level3(board_state, w.pos)
        if dist == 0:
            opponent_proximity_score += 10.0
        else:
            opponent_proximity_score += 1.0 / (dist+0.5)

    proximity = my_proximity_score - opponent_proximity_score

    # Weighted combination
    score = (8 * height_advantage) + (2 * mobility) + (6 * proximity) # Change the numbers and check which pass better
    return score

def order_moves(board, moves):
    def score_move(m):
        (w, mv, bd) = m
        r0, c0 = w.pos
        r1, c1 = mv
        h0 = board.grid[(r0, c0)].height
        h1 = board.grid[(r1, c1)].height
        build_height = board.grid[bd].height

        # Prioritize in order:
        # 1. Capping enemy towers (height 3 → dome)
        cap_bonus = 1000 if build_height == 3 else 0

        # 2. Climbing up
        climb_delta = h1 - h0

        # 3. Being on high ground
        position_height = h1

        # 4. Building high (deny opponent climbing paths)
        build_score = build_height * 10

        return cap_bonus, climb_delta, position_height, build_score

    return sorted(moves, key=score_move, reverse=True)


def evaluate_mcts(board_state, player_id):
    # for rust mcts especially
    game_config = board_state.game_config
    # 0 1 2 == p1 p2 p3
    if isinstance(player_id, int):
        root_player_id = game_config.get_player_id(player_id)
    else:
        root_player_id = player_id

    # --- 1. Tactical win/loss check ---
    my_workers = [w for w in board_state.workers if w.owner == player_id]
    opponent_workers = [w for w in board_state.workers if w.owner != player_id]


    root_player_id = game_config.get_player_id(player_id)
    # Win in one move?
    for w in my_workers:
        if w.pos is None:
            continue
        src = w.pos
        for move in legal_moves(board_state, src):
            if is_win_after_move(board_state, src, move):
                return 0.9  # "almost certain win" for root

    # 2) If any opponent can win in one from this state, make it very bad
    if opponent_can_win_in_one(board_state, root_player_id):
        return -0.9  # "almost certain loss" for root
    # --- 2. Positional heuristic ---
    height_adv = sum(
        board_state.get_cell(w.pos).height for w in my_workers
    ) - sum(
        board_state.get_cell(w.pos).height for w in opponent_workers
    )

    my_mobility = 0.0
    for w in my_workers:
        for move in legal_moves(board_state, w.pos):
            h_src = board_state.get_cell(w.pos).height
            h_dst = board_state.get_cell(move).height
            diff = h_dst - h_src
            if diff == 1:
                my_mobility += 2.0
            elif diff == 0:
                my_mobility += 1.0
            elif diff < 0:
                my_mobility += 0.5

    opp_mobility = 0.0
    for w in opponent_workers:
        for move in legal_moves(board_state, w.pos):
            h_src = board_state.get_cell(w.pos).height
            h_dst = board_state.get_cell(move).height
            diff = h_dst - h_src
            if diff == 1:
                opp_mobility += 2.0
            elif diff == 0:
                opp_mobility += 1.0
            elif diff < 0:
                opp_mobility += 0.5

    mobility = my_mobility - opp_mobility

    # Distance to nearest level 3
    level3 = [
        (x, y)
        for x in range(BOARD_SIZE)
        for y in range(BOARD_SIZE)
        if board_state.get_cell((x, y)).height == 3
    ]
    my_dist = 0.0
    for w in my_workers:
        x0, y0 = w.pos
        if level3:
            dist = min(max(abs(x - x0), abs(y - y0)) for (x, y) in level3)
        else:
            dist = BOARD_SIZE
        my_dist += (BOARD_SIZE - dist)  # closer is better

    opp_dist = 0.0
    for w in opponent_workers:
        x0, y0 = w.pos
        if level3:
            dist = min(max(abs(x - x0), abs(y - y0)) for (x, y) in level3)
        else:
            dist = BOARD_SIZE
        opp_dist += (BOARD_SIZE - dist)

    dist_sum = my_dist - opp_dist
    my_tower_ctrl = local_tower_control(board_state, my_workers)
    opp_tower_ctrl = local_tower_control(board_state, opponent_workers)
    tower_ctrl = my_tower_ctrl - opp_tower_ctrl
    # --- 3. Weighted combination ---
    raw = 8 * height_adv + 3 * mobility + 6 * dist_sum + 4* tower_ctrl
# Normalize to [-0.5, 0.5]
    v = raw / 100.0
    if v > 0.5:
        v = 0.5
    if v < -0.5:
        v = -0.5
    return v


def find_win_in_one(board, player_id):

    #Returns (worker, move_pos, build_pos) if player_id can win in one move,
    #otherwise returns None.

    workers = [w for w in board.workers if w.owner == player_id]

    for w in workers:
        if w.pos is None:
            continue
        src = w.pos
        for move in legal_moves(board, src):
            # Win condition is purely about the move, not the build
            if is_win_after_move(board, src, move):
                # After a winning move, any legal build is fine; pick first
                builds = legal_builds(board, move)
                build_pos = builds[0] if builds else move
                return w, move, build_pos

    return None

def opponent_can_win_in_one(board_state, root_player_id: str) -> bool:
    # Returns True if any opponent can win in one move against root_player_id
    game_config = board_state.game_config

    for opp_id in game_config.player_ids:
        if opp_id == root_player_id:
            continue

        for w in board_state.workers:
            if w.owner != opp_id or w.pos is None:
                continue

            src = w.pos
            for move in legal_moves(board_state, src):
                if is_win_after_move(board_state, src, move):
                    return True
    return False

def simulate_action(board, worker, move, build):
    new_board = board.clone()

    # Find worker on cloned board
    cloned_worker = None
    for w in new_board.workers:
        if w.id == worker.id:
            cloned_worker = w
            break
    if cloned_worker is None:
        return None, None

    won = move_worker(new_board, cloned_worker, move)
    if cloned_worker.pos != move:
        return None, None

    build_block(new_board, cloned_worker, build)

    return new_board, won

def evaluate_action(board, player_id, action):
    worker, move, build = action
    new_board, won = simulate_action(board, worker, move, build)

    if new_board is None:  # illegal
        return -99.0

    if won:
        return 100.0

    # Positional heuristics AFTER move
    my_workers = [w for w in new_board.workers if w.owner == player_id]
    opponent_workers = [w for w in new_board.workers if w.owner != player_id]

    # --- height advantage ---
    my_height = sum(new_board.get_cell(w.pos).height for w in my_workers)
    opp_height = sum(new_board.get_cell(w.pos).height for w in opponent_workers) /2.0
    height_adv = my_height - opp_height

    mobility = mob_score(my_workers, new_board) - (mob_score(opponent_workers, new_board) /2.0)

    # --- proximity to level 3 ---
    def proximity_score(w_list):
        s = 0.0
        for w in w_list:
            d = distance_to_nearest_level3(new_board, w.pos)
            s += 10.0 if d == 0 else 1.0 / (d + 0.5)
        return s

    proximity = proximity_score(my_workers) - (proximity_score(opponent_workers) /2.0)

    # Weighted sum
    score = 15 * height_adv + 9 * mobility + 18 * proximity
    return score

def detailed_eval(board_state, player_id, action)->Dict[str, Any]:
    worker, move, build = action

    new_board, won = simulate_action(board_state, worker, move, build)

    if new_board is None:
        return {"illegal_move": True, "total_score": -99.0}

    if won:
        return {"immediate_win": True, "total_score": 100.0}

    my_workers = [w for w in new_board.workers if w.owner == player_id]
    opponent_workers = [w for w in new_board.workers if w.owner != player_id]

    categories={}

    categories["blocks_opponent_win"] = False
    for opp_w in opponent_workers:
        for opp_move in legal_moves(board_state, opp_w.pos):
            if board_state.get_cell(opp_move).height == 3:
                if opp_move == build:
                    categories["blocks_opponent_win"] = True
                    break

    categories["opponent_can_win"] = False
    for opp_w in opponent_workers:
        for opp_move in legal_moves(new_board, opp_w.pos):
            if new_board.get_cell(opp_move).height == 3:
                categories["opponent_can_win"] = True
                break

    my_height = sum(new_board.get_cell(w.pos).height for w in my_workers)
    opponent_height = sum(new_board.get_cell(w.pos).height for w in opponent_workers) / 2
    categories["height_advantage"] = (my_height - opponent_height) * 8

    my_mobility = mob_score(my_workers, new_board)
    opponent_mobility = mob_score(opponent_workers, new_board) / 2
    categories["mobility"] = (my_mobility - opponent_mobility) * 2

    # Proximity to level 3
    my_proximity_score = 0.0
    for w in my_workers:
        dist = distance_to_nearest_level3(new_board, w.pos)
        if dist == 0:
            my_proximity_score += 10.0
        else:
            my_proximity_score += 1.0 / (dist + 0.5)

    opponent_proximity_score = 0.0
    for w in opponent_workers:
        dist = distance_to_nearest_level3(new_board, w.pos)
        if dist == 0:
            opponent_proximity_score += 10.0
        else:
            opponent_proximity_score += 1.0 / (dist + 0.5)

    categories["proximity"] = (my_proximity_score - (opponent_proximity_score /2)) * 6

    # Tower control
    my_tower_control = local_tower_control(new_board, my_workers)
    opp_tower_control = local_tower_control(new_board, opponent_workers) / 2
    categories["tower_control"] = (my_tower_control - opp_tower_control) * 2

    # Calculate total score
    categories["total_score"] = sum(
        v for k, v in categories.items() if isinstance(v, (int, float)) and k != "total_score")

    return categories