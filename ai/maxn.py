from typing import Tuple, List
from ai.heuristics import evaluate
from game.config import GameConfig
from game.moves import move_worker, build_block
from game.rules import legal_moves, legal_builds
from functools import lru_cache

# infinity const for evaluation win/lose
INF = 10 ** 9
# the type for ai's action (worker, move, build)
Action = Tuple[object, Tuple[int, int], Tuple[int, int]]

TT = {}


def maxn(board, depth, player_index, game_config, stats):
    stats.bump()
    key = (hash(board), depth, player_index)
    if key in TT:
        stats.tt_hits += 1
        return TT[key]

    # terminal node reached or depth limit
    if depth == 0 or board.game_over():
        payoff = [evaluate(board, pid) for pid in game_config.player_ids]
        TT[key] = (payoff, None)
        return payoff, None

    best_action = None
    best_vector = None

    player_id = game_config.get_player_id(player_index)
    actions = generate_actions(board, player_id)
    actions = order_moves(board, actions)

    for action in actions:
        worker, move, build = action
        new_board = board.clone()

        # find worker in new_board
        new_worker = find_worker_by_id(new_board, worker.id)
        if new_worker is None:
            print(f"[DEBUG] Worker id {worker.id} not found in cloned board at depth {depth}")
            continue

        won = move_worker(new_board, new_worker, move)
        if won:
            payoff = [0] * len(game_config.player_ids)
            payoff[player_index] = INF
            TT[key] = (payoff, action)
            return payoff, action

        build_block(new_board, new_worker, build)

        # Get next player using modular rotation
        next_index = (player_index + 1) % len(game_config.player_ids)
        child_vector, _ = maxn(new_board, depth - 1, next_index, game_config, stats)

        # Select the child that maximizes the current player's value
        if best_vector is None or child_vector[player_index] > best_vector[player_index]:
            best_vector = child_vector
            best_action = action

    if best_action is None:
        print(f"[DEBUG] No best action found at depth {depth} for player {player_index}")

    TT[key] = (best_vector, best_action)

    return best_vector, best_action

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
        r0, c0 = w.pos;
        r1, c1 = mv
        h0 = board.grid[(r0, c0)].height;
        h1 = board.grid[(r1, c1)].height
        delta = h1 - h0  # climbing is good in Santorini
        # prefer building next to our worker and capping enemy’s towers later
        return (delta, h1, -(board.grid[bd].height))

    return sorted(moves, key=score_move, reverse=True)
