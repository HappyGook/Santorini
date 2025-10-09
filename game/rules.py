from typing import List, Tuple
from game.models import Coord, MAX_LEVEL, DOME_LEVEL
from game.board import Board
from collections import Counter
from game.models import Worker
from game.config import MAX_WORKERS_PER_PLAYER


def can_place_worker(board: "Board", owner: str, pos: Tuple[int, int]) -> bool:
    """Pure check: is initial placement at pos legal for owner on this board?"""
    # on-board & existing cell?
    if pos not in board.grid:
        return False

    cell = board.grid[pos]


    if cell.worker_id is not None:
        return False

    # at most 2 workers per owner
    owner_count = sum(1 for w in board.workers if w.owner == owner)
    if owner_count >= MAX_WORKERS_PER_PLAYER:
        return False

    return True
def can_move(board: Board, src: Coord, dst: Coord) -> bool:
    """A move is legal iff:
      - dst is 1cell away to src (8 directions)
      - dst is not occupied
      - dst is not a dome (height != DOME_LEVEL)
      - climb <= +1"""

    if dst not in board.neighbors(src):
        return False
    if board.is_occupied(dst):
        return False
    if board.get_cell(dst).height >= DOME_LEVEL:
        return False

    h_src = board.get_cell(src).height
    h_dst = board.get_cell(dst).height

    if h_dst > h_src + 1:
        return False
    return True

def can_build(board: Board, stand_pos: Coord, target: Coord) -> bool:
    """ A build is legal iff:
      - target is adjacent to the worker's current position (after moving)
      - target is not occupied
      - target is not a dome
    """

    if target not in board.neighbors(stand_pos):
        return False
    if board.is_occupied(target):
        return False
    if board.get_cell(target).height >= DOME_LEVEL:
        return False
    return True

def is_win_after_move(board: Board, src: Coord, dst: Coord) -> bool:
    """A player wins by moving one of their workers onto a level 3 space."""
    h_src = board.get_cell(src).height
    h_dst = board.get_cell(dst).height

    return h_src < MAX_LEVEL and h_dst == MAX_LEVEL

def legal_moves(board: Board, src: Coord) -> List[Coord]:
    """Return all legal move destinations from src."""
    return [p for p in board.neighbors(src) if can_move(board, src, p)]

def legal_builds(board: Board, stand_pos: Coord) -> List[Coord]:
    """Return all legal build positions from stand_pos."""
    return [p for p in board.neighbors(stand_pos) if can_build(board, stand_pos, p)]

def player_has_moves(board: Board, player: str) -> bool:
    """Return True if the player has any legal moves + builds left."""
    # Get all workers of player on board
    workers = [w for w in board.workers if w.owner == player]
    for worker in workers:
        moves = legal_moves(board, worker.pos)
        for move_pos in moves:
            builds = legal_builds(board, move_pos)
            if builds:
                return True
    return False

def all_legal_actions(board: Board, player: str) -> List[Tuple[Coord, Coord]]:
    """
    Return all legal (move_pos, build_pos) pairs for the player's workers.
    Each pair represents a move followed by a build.
    """
    actions = []
    workers = [w for w in board.workers if w.owner == player]
    for worker in workers:
        for move_pos in legal_moves(board, worker.pos):
            for build_pos in legal_builds(board, move_pos):
                actions.append((move_pos, build_pos))
    return actions