from typing import List, Tuple
from models import DOME_LEVEL
from board import Board
from models import Worker

Coord = Tuple[int, int]
Move = Tuple[Worker, Coord]  # (worker, move_to)
Build = Tuple[Worker, Coord]  # (worker, build_at)

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),          (0, 1),
    (1, -1),  (1, 0), (1, 1),
]

def get_legal_moves(board: Board, player: str) -> List[Move]:
    legal_moves: List[Move] = []

    for worker in board.workers:
        if worker.owner != player or worker.pos is None:
            continue

        src = worker.pos
        src_height = board.get_cell(src).height

        for dr, dc in DIRECTIONS:
            dst = (src[0] + dr, src[1] + dc)
            if not board.in_bounds(dst):
                continue
            dst_cell = board.get_cell(dst)
            if dst_cell.worker_id is not None:
                continue
            if dst_cell.height > src_height + 1:
                continue
            if dst_cell.height >= DOME_LEVEL:
                continue

            legal_moves.append((worker, dst))

    return legal_moves

def get_legal_builds(board: Board, worker: Worker, dst: Coord) -> List[Build]:
    legal_builds: List[Build] = []
    for br, bc in DIRECTIONS:
        build = (dst[0] + br, dst[1] + bc)
        if not board.in_bounds(build):
            continue
        build_cell = board.get_cell(build)
        if build_cell.worker_id is not None:
            continue
        if build_cell.height >= DOME_LEVEL:
            continue

        legal_builds.append((worker, build))

    return legal_builds

def move_worker(board: Board, player: str, move: Move) -> bool:
    legal_moves = get_legal_moves(board, player)
    if not legal_moves:
        raise RuntimeError(f"Player {player} has no legal moves.")
    if move not in legal_moves:
        raise ValueError("Illegal move")

    worker, dst = move
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

def build_block(board: Board, build: Build) -> None:
    worker, build_pos = build
    build_cell = board.get_cell(build_pos)
    build_cell.height += 1

    # Switch turn
    board.current_player = "P2" if board.current_player == "P1" else "P1"