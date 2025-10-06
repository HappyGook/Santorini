import copy
from typing import Dict, List, Optional
from game.models import Cell, Coord, BOARD_SIZE, Worker

class Board:
    def __init__(self, workers: Optional[ List[Worker]] = None) -> None:
        self.grid: Dict[Coord, Cell] = {
            (r, c): Cell() for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
        }
        self.workers: List[Worker] = workers or []
        self.current_player: str = "P1"

    def in_bounds(self, pos: Coord) -> bool:
        """Return True if pos on the board."""
        row, col = pos
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def get_cell(self, pos: Coord) -> Cell:
        """Return the Cell at pos; raises if out  bounds."""
        if not self.in_bounds(pos):
            raise ValueError("Position out of bounds")
        return self.grid[pos]

    def is_occupied(self, pos: Coord) -> bool:
        """True if a worker is on pos."""
        return self.get_cell(pos).worker_id is not None

    def neighbors(self, pos: Coord) -> List[Coord]:
        """8-directional neighbors within bounds."""
        r, c = pos
        deltas = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]
        out: List[Coord] = []
        for dr, dc in deltas:
            p = (r + dr, c + dc)
            if self.in_bounds(p):
                out.append(p)
        return out

    def clone(self) -> "Board":
        return copy.deepcopy(self)

    def print_board(self) -> None:
        """
        Print a readable representation of the board grid.
        For each cell, show its coordinates, height, and worker (if any).
        """
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cell = self.grid[(r, c)]
                print(f"({r},{c}): h={cell.height}, w={cell.worker_id}")
            print()  # Blank line between rows for readability
