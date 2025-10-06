from typing import Tuple, Optional, List
from ai.heuristics import evaluate
from game.moves import move_worker, build_block
from game.rules import legal_moves, legal_builds

# infinity const for evaluation win/lose
INF = 10 ** 9
# the type for ai's action (worker, move, build)
Action = Tuple[object, Tuple[int, int], Tuple[int, int]]

def minimax(board, depth: int, player_id: str, max_player_id: str, alpha=-INF, beta=INF, maximizing=True) -> Tuple[
    int, Optional[Action]]:
    """
    Alpha-beta minimax strategy
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
    value = -INF if maximizing else INF

    for action in actions:
        worker, move, build = action
        new_board = board.clone()

        # find worker in new_board
        new_worker = find_worker_by_id(new_board, worker.id)
        if new_worker is None:
            print(f"[DEBUG] Worker id {worker.id} not found in cloned board at depth {depth}")
            continue

        print(f"[DEBUG] Depth {depth}: trying action {worker.id} move {move} build {build}")

        won = move_worker(new_board, new_worker, move)
        print(f"[DEBUG] after move: worker {worker.id} at {new_worker.pos}, dst.height={new_board.get_cell(move).height}, dst.worker_id={new_board.get_cell(move).worker_id}")

        if won:
            return (INF if maximizing else -INF), action

        build_block(new_board, new_worker, build)
        print(f"[DEBUG] after build: cell {build} height={new_board.get_cell(build).height}")

        # recursively evaluate
        score, _ = minimax(new_board, depth - 1, other(player_id), max_player_id, alpha, beta, not maximizing)
        print(f"[DEBUG] Depth {depth} player {player_id} action {worker.id} move {move} build {build} "
              f"score={score} value={value} alpha={alpha} beta={beta}")

        # update best action based on maximizing/minimizing
        if maximizing:
            if score > value:
                value = score
                best_action = action
            alpha = max(alpha, value)
        else:
            if score < value:
                value = score
                best_action = action
            beta = min(beta, value)

        # alpha-beta pruning
        if beta <= alpha:
            print(f"[DEBUG] Depth {depth}: alpha-beta cutoff")
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

def other(player_id):
    return "P1" if player_id == "P2" else "P2"

def find_worker_by_id(board, worker_id):
    return next((w for w in board.workers if w.id == worker_id), None)