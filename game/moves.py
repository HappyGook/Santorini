from typing import Tuple
from game.board import Board
from game.models import Worker
from game.rules import can_place_worker
Coord = Tuple[int, int]

def move_worker(board: Board, worker: Worker, dst: Coord) -> bool:
    src = worker.pos

    # Update grid cells
    board.get_cell(src).worker_id = None
    dst_cell = board.get_cell(dst)
    dst_cell.worker_id = worker.id

    # Update worker
    worker.pos = dst

    # Check win condition
    if dst_cell.height == 3:
        return True

    return False

def build_block(board: Board, worker: Worker, build_pos: Coord) -> None:
    build_cell = board.get_cell(build_pos)
    build_cell.height += 1

    from typing import Tuple


def place_worker(board: "Board", worker_id: str, owner: str, pos: Tuple[int, int]) -> bool:

    if not can_place_worker(board, owner, pos):
        return False

    cell = board.grid[pos]
    w = Worker(id=worker_id, owner=owner, pos=pos)
    board.workers.append(w)
    cell.worker_id = worker_id
    return True
