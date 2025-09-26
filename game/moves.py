from typing import Tuple
from board import Board
from models import Worker

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

    # Switch turn
    board.current_player = "P2" if board.current_player == "P1" else "P1"