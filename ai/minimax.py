from typing import Tuple, Optional, List
from ai.heuristics import evaluate
from game.moves import move_worker, build_block
from game.rules import legal_moves, legal_builds  # not all_legal_actions

# inifinity const for evaluation win/lose
INF = 10**9
# the type for ai's action (worker, move, build)
Action = Tuple[object, Tuple[int,int], Tuple[int,int]]

def minimax(board, depth: int, player_id: str, max_player_id: str, alpha=-INF, beta=INF, maximizing=True) -> Tuple[int, Optional[Action]]:
    """
       alpha-beta minimax strategy (i hope this works)
       :param board: current board state
       :param depth: how deep we want to search
       :param player_id: whose turn it is at this node
       :param max_player_id: the player we evaluate from/for
    """
    # no actions -> losing state for current player
    actions = generate_actions(board, player_id)
    if depth == 0 or not actions:
        return evaluate(board, max_player_id), None

    best_action = None

    if maximizing:
        value = -INF
        for action in actions:
            worker, move, build = action
            new_board = board.clone()
            # find worker in new_board
            try:
                new_worker = next(w for w in new_board.workers if w.id == worker.id)
            except StopIteration:
                # Worker not found in clone? fallback to invalid action skip
                print(f"[DEBUG] Worker id {worker.id} not found in cloned board at depth {depth}")
                continue
            won = move_worker(new_board, new_worker, move)
            if won:
                return INF, action
            build_block(new_board, new_worker, build)

            # call the function again, with depth decreased, and this time we minimize (coz other player's move)
            score, _ = minimax(new_board, depth-1, other(player_id), max_player_id, alpha, beta, False)
            if score > value:
                value = score
                best_action = action
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        if best_action is None and actions:
            print(f"[WARN] No valid actions selected at depth {depth} for {player_id}, "
                  f"actions count={len(actions)}")
            print("Old board:\n")
            board.print_board()
            print("New board:\n")
            new_board.print_board()
        return value, best_action
    else:
        value = INF
        for action in actions:
            worker, move, build = action
            new_board = board.clone()
            try:
                new_worker = next(w for w in new_board.workers if w.id == worker.id)
            except StopIteration:
                print(f"[DEBUG] Worker id {worker.id} not found in cloned board at depth {depth}")
                continue
            won = move_worker(new_board, new_worker, move)
            if won:
                return -INF, action  # opponent winning is bad for root?
            build_block(new_board, new_worker, build)

            score, _ = minimax(new_board, depth-1, other(player_id), max_player_id, alpha, beta, True)
            if score < value:
                value = score
                best_action = action
            beta = min(beta, score)
            if beta <= alpha:
                break
        if best_action is None and actions:
            print(f"[WARN] No valid actions selected at depth {depth} for {player_id}, "
                  f"actions count={len(actions)}")
            print("Old board:\n")
            board.print_board()
            print("New board:\n")
            new_board.print_board()
        return value, best_action

def generate_actions(board, player_id) -> List[Action]:
    actions = []
    workers = [w for w in board.workers if w.owner == player_id]
    for worker in workers:
        for move in legal_moves(board, worker.pos):
            for build in legal_builds(board, move):
                actions.append((worker, move, build))
    return actions

def other(player_id): return "P1" if player_id=="P2" else "P2"