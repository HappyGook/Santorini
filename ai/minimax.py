from typing import Tuple, Optional, List
from ai.heuristics import evaluate
from game.config import GameConfig
from game.moves import move_worker, build_block
from game.rules import legal_moves, legal_builds
from functools import lru_cache

# infinity const for evaluation win/lose
INF = 10 ** 9
# the type for ai's action (worker, move, build)
Action = Tuple[object, Tuple[int, int], Tuple[int, int]]

TT ={}
def minimax(board, depth, player_index, max_player_index, stats, maximizing=True):
    stats.bump()
    key = (hash(board), depth, player_index, maximizing)
    if key in TT:
        stats.tt_hits += 1
        return TT[key]

    """
    Alpha-beta minimax strategy
    :param board: current board state
    :param depth: how deep we want to search
    :param player_index: current player as integer (0, 1, 2...)
    :param max_player_index: the player we're maximizing for
    """
    player_id = board.game_config.get_player_id(player_index)
    actions = generate_actions(board, player_id)
    actions = order_moves(board, actions)

    if depth == 0 or not actions:
        max_player_id = board.game_config.get_player_id(max_player_index)
        score = evaluate(board, max_player_id) 
        TT[key] = (score, None)
        return score, None

    best_action = None
    value = -INF if maximizing else INF

    for action in actions:
        worker, move, build = action
        new_board = board.clone()

        # find worker in new_board
        new_worker = find_worker_by_id(new_board, worker.id)
        if new_worker is None:
            print(f"[DEBUG] Worker id {worker.id} not found in cloned board at depth {depth}")
            continue

        #print(f"[DEBUG] Depth {depth}: trying action {worker.id} move {move} build {build}")

        won = move_worker(new_board, new_worker, move)
        #print(f"[DEBUG] after move: worker {worker.id} at {new_worker.pos}, dst.height={new_board.get_cell(move).height}, dst.worker_id={new_board.get_cell(move).worker_id}")

        if won:
            leaf_score = INF if maximizing else -INF
            TT[key] = (leaf_score, action)  # cache winning child
            return leaf_score, action
           

        build_block(new_board, new_worker, build)
        #print(f"[DEBUG] after build: cell {build} height={new_board.get_cell(build).height}")

        # Get next player using modular rotation
        next_player_index = board.game_config.next_player_index(player_index)

        # For 3 players, we need to determine if next player is maximizing
        next_maximizing = (next_player_index == max_player_index)

        # recursively evaluate
        score, _ = minimax(new_board, depth - 1, next_player_index, max_player_index, stats,
                            next_maximizing)

        #print(f"[DEBUG] Depth {depth} player {player_id} action {worker.id} move {move} build {build} "
        #      f"score={score} value={value}")

        
        # update best action based on maximizing/minimizing
        if maximizing:
            if score > value:
                value = score
                best_action = action
            
        else:
            if score < value:
                value = score
                best_action = action
            

      

    if best_action is None and actions:
        print(f"[WARN] No valid actions selected at depth {depth} for {player_id}, "
              f"actions count={len(actions)}")
        print("Old board:\n")
        board.print_board()
        print("New board:\n")
        new_board.print_board()
        best_action = actions[0]
    TT[key] = (value, best_action) 
    return value, best_action

    

def generate_actions(board, player_id) -> List[Action]:
    actions = []
    workers = [w for w in board.workers if w.owner == player_id]
    for worker in workers:
        for move in legal_moves(board, worker.pos):
            for build in legal_builds(board, move):
                actions.append((worker, move, build))
    return actions

def next_player_index(current_index: int, num_players: int) -> int:
    """Simple modular rotation - can be moved to config"""
    return (current_index + 1) % num_players

def next_player(player_id: str, game_config: GameConfig) -> str:
    current_index = game_config.get_player_index(player_id)
    next_index = game_config.next_player_index(current_index)
    return game_config.get_player_id(next_index)


def find_worker_by_id(board, worker_id):
    return next((w for w in board.workers if w.id == worker_id), None)

class SearchStats:
    def __init__(self):
        self.nodes = 0
        self.tt_hits = 0
    def bump(self): self.nodes += 1

def order_moves(board, moves):
    # moves: list[(worker, move_to, build_to)]
    def score_move(m):
        (w, mv, bd) = m
        r0, c0 = w.pos; r1, c1 = mv
        h0 = board.grid[(r0,c0)].height; h1 = board.grid[(r1,c1)].height
        delta = h1 - h0      # climbing is good in Santorini
        # prefer building next to our worker and capping enemy’s towers later
        return (delta, h1, -(board.grid[bd].height))
    return sorted(moves, key=score_move, reverse=True)
