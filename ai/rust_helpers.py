from typing import List, Tuple
from copy import deepcopy
from game.models import BOARD_SIZE
from game.rules import legal_moves, legal_builds, is_win_after_move
from game.moves import move_worker, build_block
from ai.heuristics import evaluate_mcts
from game.models import MAX_LEVEL
Action = Tuple[object, Tuple[int, int], Tuple[int, int]]


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
        # legal move positions for this worker
        for move in legal_moves(board, src):
            # simulate move on a copy
            temp_board = deepcopy(board)
            temp_worker = next(w for w in temp_board.workers if w.id == worker.id)
            move_worker(temp_board, temp_worker, move)

            # legal builds from the new position
            for build in legal_builds(temp_board, move):
                actions.append((worker, move, build))

    return actions



def apply_action(board, action: Action):
    # return a new board state after applying the given action

    worker_id, move_pos, build_pos = action

    new_board = deepcopy(board)
    
    # find the worker object in the copied board
    target_id = worker_id.id

    worker = None
    for w in new_board.workers:
        if w.id == target_id:
            worker = w
            break

    if worker is None:
        # Defensive: avoid StopIteration and give a clear error
        raise RuntimeError(f"apply_action: no worker with id {target_id!r} on new_board")
    # apply move and build using your existing game.moves functions
    move_worker(new_board, worker, move_pos)
    build_block(new_board, worker, build_pos)
    
    return new_board

def is_terminal(board, player_index:int) ->bool:
    #true if the game is over (win/loss)

    for worker in board.workers:
        src =  worker.pos

        for worker in board.workers:
            if worker.pos is None:
                continue
        cell = board.get_cell(worker.pos)
        if cell.height == MAX_LEVEL:
            return True
    return False


def evaluate_board(board, root_player_index: int) -> float:
#Evaluate the board state 

    return float(evaluate_mcts(board, root_player_index))