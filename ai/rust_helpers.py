from typing import List, Tuple, Optional
from copy import deepcopy
from game.models import BOARD_SIZE
from game.rules import legal_moves, legal_builds, player_has_moves
from game.moves import move_worker, build_block
from ai.heuristics import evaluate_mcts
from game.models import MAX_LEVEL
Action = Tuple[int, Tuple[int, int], Tuple[int, int]]


def list_actions(board, player_index: int) -> List[Action]:
    
   # Return all legal (worker_id, move_pos, build_pos) actions for this player.
    
    actions: List[Action] = []

    # Map player_index (0,1,...) -> "P1","P2",...
    player_id = board.game_config.get_player_id(player_index)

    
    workers = [
        w for w in board.workers
        if w.owner == player_id and w.pos is not None
    ]

    for worker in workers:
        src = worker.pos

        # legal move positions for this worker (on the original board)
        for move in legal_moves(board, src):
            # legal build positions from the new position
            for build in legal_builds(board, move):
                # store only worker.id, not the whole worker object
                actions.append((worker.id, move, build))

    return actions


def apply_action(board, action: Action):
    # return a new board state after applying the given action

    worker_id, move_pos, build_pos = action  # worker_id is a str like "P2A"

    new_board = deepcopy(board)

    # find the worker object in the copied board
    worker = None
    for w in new_board.workers:
        if w.id == worker_id:
            worker = w
            break

    if worker is None:
        raise RuntimeError(f"apply_action: no worker with id {worker_id!r} on new_board")

    move_worker(new_board, worker, move_pos)
    build_block(new_board, worker, build_pos)

    return new_board


def terminal_value(board, root_player_index: int) -> float | None:
    """
    Return:
      +1.0 if root player has already won,
      -1.0 if someone else has already won,
      0.0 if no moves left for root player (loss),
      None otherwise (non-terminal).
    """
    # 1) someone stands on level 3
    winner_owner = None
    for w in board.workers:
        if w.pos is None:
            continue
        cell = board.get_cell(w.pos)
        if cell.height == MAX_LEVEL:
            winner_owner = w.owner
            break

    root_player_id = board.game_config.get_player_id(root_player_index)

    if winner_owner is not None:
        if winner_owner == root_player_id:
            return 1.0
        else:
            return -1.0

    # 2) no moves left for root player -> treat as loss
    if not player_has_moves(board, root_player_id):
        return -1.0

    return None


def evaluate_board(board, root_player_index: int) -> float:
#Evaluate the board state 

    return float(evaluate_mcts(board, root_player_index))

def terminal_value(board, root_player_index: int) -> Optional[float]:
    """
    +1.0 if the root player already stands on level 3,
    -1.0 if any OTHER player already stands on level 3,
     None if not terminal.
    """
    for w in board.workers:
        if w.pos is None:
            continue
        if board.get_cell(w.pos).height == MAX_LEVEL:
            winner_idx = board.game_config.get_player_index(w.owner)
            return 1.0 if winner_idx == root_player_index else -1.0
    return None