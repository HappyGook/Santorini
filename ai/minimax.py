from typing import Tuple, Optional, List

from ai.heuristics import evaluate
from game.moves import move_worker, build_block
from game.rules import all_legal_actions, is_win_after_move

# inifinity const for evaluation win/lose
INF = 10**9

# the type for ai's action (worker, move, build)
Action = Tuple[object, tuple[int, int], tuple[int, int]]

def minimax(board, depth: int, player_id: str, max_player_id: str,
            alpha=-INF, beta=INF, maximizing=True) -> Tuple[int, Optional[Action]]:
    """
    alpha-beta minimax strategy (i hope this works)
    :param board: current board state
    :param depth: how deep we want to search
    :param player_id: whose turn it is at this node
    :param max_player_id: the player we evaluate from/for
    """
    actions = generate_actions(board, player_id)

    # no actions -> losing state for current player
    if depth == 0 or not actions:
        return evaluate(board, max_player_id), None

    if maximizing:
        value = -INF
        for action in actions:
            worker, move, build = action

            prev_state = board.clone() # get the board state
            move_worker(board, worker, move)
            if is_win_after_move(prev_state, worker.pos, move):
                board = prev_state
                return INF, action
            build_block(board, worker, build)

            # call the function again, with depth decreased, and this time we minimize (coz other player's move)
            score,_=minimax(board, depth-1, other(player_id), max_player_id, alpha, beta, maximizing = False)

            board = prev_state

            if score > value:
                value = score
                best_action = action
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return value, best_action
    else:
        value = INF
        for action in actions:
            worker, move, build = action

            prev_state = board.clone() # get the board state
            move_worker(board, worker, move)
            if is_win_after_move(prev_state, worker.pos, move):
                board = prev_state
                return INF, action
            build_block(board, worker, build)

            score,_=minimax(board, depth-1, other(player_id), max_player_id, alpha, beta, maximizing = True)
            board = prev_state

            if score < value:
                value = score
                best_action = action
            beta = min(beta, score)
            if beta <= alpha:
                break
        return value, best_action

def generate_actions(board, player_id) -> List[Action]:
    """Return all legal (worker, move, build) triples for the player."""
    actions = []
    workers = [w for w in board.workers if w.owner == player_id]
    for worker in workers:
        for move, build in all_legal_actions(board, player_id):
            actions.append((worker, move, build))
    return actions


def other(player_id: str) -> str:
    return "P1" if player_id == "P2" else "P2"