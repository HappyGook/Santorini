from ai.heuristics import evaluate, order_moves
from game import rules
from game.moves import move_worker, build_block, find_worker_by_id
from game.rules import all_legal_actions
from ml.inference import ml_inference

# infinity const for evaluation win/lose
INF = 10 ** 9

TT = {}


def maxn(board, depth, player_index, game_config, stats,  ml_model=None, ancestor_index=None, ancestor_best=None):
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
    if depth == 0 or rules.game_over(board):
        payoff=[]
        # Use model if passed
        if ml_model is not None:
            for i, pid in enumerate(game_config.player_ids):
                value,_=ml_inference(board, i, ml_model, stats)
                if value is not None:
                    value = max(-100, min(100, value * 100))
                    payoff.append(value)
                else:
                    payoff.append(evaluate(board, pid))
        else:
            payoff = [evaluate(board, pid) for pid in game_config.player_ids]

        TT[key] = (payoff, None)
        return payoff, None

    best_action = None
    best_vector = None

    player_id = game_config.get_player_id(player_index)
    actions = all_legal_actions(board, player_id)
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

            # Add depth penalty/bonus - only modify once
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
        child_vector, _ = maxn(new_board, depth - 1, next_index, game_config, stats, ml_model,
                               ancestor_index if ancestor_index is not None else player_index,
                               ancestor_best if ancestor_best is not None else best_vector[
                                   player_index] if best_vector else -INF)

        # Select the child that maximizes the current player's value
        if best_vector is None or child_vector[player_index] > best_vector[player_index]:
            best_vector = child_vector
            best_action = action

        # Pruning check: if this branch is worse for the ancestor than their current best,
        # they won't choose this path, so we can prune
        if ancestor_index is not None and ancestor_index != player_index:
            if best_vector[ancestor_index] <= ancestor_best:
                break

    if best_action is None:
        print(f"[DEBUG] No best action found at depth {depth} for player {player_index}")

    TT[key] = (best_vector, best_action)
    return best_vector, best_action

class SearchStats:
    def __init__(self):
        self.nodes = 0
        self.tt_hits = 0

    def bump(self):
        self.nodes += 1