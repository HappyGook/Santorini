from typing import List
from .models import Coord, MAX_LEVEL, DOME_LEVEL
from .board import Board

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

    if h_dst > h_src + 1: #climb more than 1 nicht erlaubt
        return False
    return True

def can_build(board: Board, stand_pos: Coord, target: Coord) -> bool:
    """ A build is legal iff:
      - target is adjacent to the worker's current position (after moving)
      - target is not occupied
      - target is nota dome 
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

#not needed return all legal move/ build positions

def legal_moves(board: Board, src: Coord) -> List[Coord]:
    #all dst cell from src
    return [p for p in board.neighbors(src) if can_move(board, src,p)]

def legal_builds(board: Board, stand_pos: Coord) -> List[Coord]:
    return [p for p in board.neighbors(stand_pos) if can_build(board, stand_pos, p)]