"""
This is a heuristics file: evaluation of the moves in the game.
For now I have made 3 factors:
    1. Height advantage
    2. Mobility
    3. Proximity to level 3
"""

from game.board import BOARD_SIZE
from game.rules import legal_moves

def distance_to_nearest_level3(board_state, pos):
    level3_positions = [
        (x, y)
        for x in range(BOARD_SIZE)
        for y in range(BOARD_SIZE)
        if board_state.get_cell((x, y)).height == 3
    ]
    if not level3_positions:
        return BOARD_SIZE * 2  # max possible distance on board TODO: diagonals

    x0, y0 = pos
    distances = [abs(x - x0) + abs(y - y0) for (x, y) in level3_positions]
    return min(distances)

def evaluate(board_state, player_id):
    my_workers = [w for w in board_state.workers if w.owner == player_id]
    opponent_workers = [w for w in board_state.workers if w.owner != player_id]

    # Height advantage: sum of my workers' cell levels minus opponent's
    my_height = sum(board_state.get_cell(w.pos).height for w in my_workers)
    opponent_height = sum(board_state.get_cell(w.pos).height for w in opponent_workers)
    height_advantage = my_height - opponent_height

    # Mobility: sum of my legal moves minus opponent's
    my_mobility = sum(len(legal_moves(board_state, w.pos)) for w in my_workers)
    opponent_mobility = sum(len(legal_moves(board_state, w.pos)) for w in opponent_workers)
    mobility = my_mobility - opponent_mobility

    # Proximity: for each worker distance to nearest level-3 cell (reward being closer)
    my_proximity_score = 0
    for w in my_workers:
        dist = distance_to_nearest_level3(board_state, w.pos)
        if dist == 0:
            my_proximity_score += 10  # bonus for being on level 3
        else:
            my_proximity_score += 1 / dist

    opponent_proximity_score = 0
    for w in opponent_workers:
        dist = distance_to_nearest_level3(board_state, w.pos)
        if dist == 0:
            opponent_proximity_score += 10
        else:
            opponent_proximity_score += 1 / dist

    proximity = my_proximity_score - opponent_proximity_score

    # Weighted combination
    score = (5 * height_advantage) + (3 * mobility) + (4 * proximity) # Change the numbers and check which pass better
    return score
