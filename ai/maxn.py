from typing import Tuple, List
from ai.heuristics import evaluate, order_moves
from game.config import GameConfig
from game.moves import move_worker, build_block
from game.rules import legal_moves, legal_builds

# infinity const for evaluation win/lose
INF = 10 ** 9
# the type for ai's action (worker, move, build)
Action = Tuple[object, Tuple[int, int], Tuple[int, int]]

TT = {}


def maxn(board, depth, player_index, game_config, stats, ancestor_index=None, ancestor_best=None,):
    """
        Max^n with deep pruning.
        Each player maximizes their own component.
        ancestor_index: index of the ancestor player whose value we track for pruning
        ancestor_best: best value found so far for that ancestor
    """
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

    if not actions:
        n = len(game_config.player_ids)
        payoff = [0] * n
        payoff[player_index] = -INF
        TT[key] = (payoff, None)
        return payoff, None
    
    
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
            n = len(game_config.player_ids)
            payoff = [-INF] * n
            payoff[player_index] = INF

            payoff[player_index] -= depth
            # Add depth penalty/bonus
            for i in range(n):
                if i != player_index:
                    payoff[i] += depth  # Losing later is slightly better
                else:
                    payoff[i] -= depth  # Winning sooner is better
            TT[key] = (payoff, action)
            print(f"[WIN] Player {player_index} wins at depth {depth} after action {action}")
            return payoff, action

        build_block(new_board, new_worker, build)

        # Get next player using modular rotation
        next_index = (player_index + 1) % len(game_config.player_ids)
        child_vector, _ = maxn(new_board, depth - 1, next_index, game_config, stats,
                               ancestor_index if ancestor_index is not None else player_index,
                               ancestor_best if ancestor_best is not None else -INF)

        # Select the child that maximizes the current player's value
        if best_vector is None or child_vector[player_index] > best_vector[player_index]:
            best_vector = child_vector
            best_action = action

        if best_action is None or child_vector[player_index] > best_vector[player_index]:
            best_vector = child_vector
            best_action = action
            print(f"[DEBUG] No best action found at depth {depth} for player {player_index}")

            # pruning check
            if ancestor_index is not None and best_vector[ancestor_index] < ancestor_best:
                # ancestor would not choose this path — prune
                break


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