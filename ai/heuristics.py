"""
This is a heuristics file: evaluation of the moves in the game.
For now I have made 3 factors:
    1. Height advantage
    2. Mobility
    3. Proximity to level 3
    4. Tactical win in one(me) /or lose (op)
"""

from game.board import BOARD_SIZE
from game.rules import legal_moves

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

    # --- mobility weighted from up/flat/down
    def mob_score(workers):
        s = 0.0
        for w in workers:
            currentH = board_state.get_cell(w.pos).height
            moves = legal_moves(board_state, w.pos)
            up = flat = down = 0
            for dst in moves:
                dstH = board_state.get_cell(dst).height
                if dstH > currentH:   up += 1
                elif dstH == currentH: flat += 1
                else:            down += 1
            s += 2.0 * up + 1.0 * flat + 0.5 * down
        return s

    my_mobility = mob_score(my_workers)
    opponent_mobility = mob_score(opponent_workers)
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
    score = (5 * height_advantage) + (3 * mobility) + (4 * proximity) # Change the numbers and check which pass better
    return score
