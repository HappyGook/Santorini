import copy
from typing import Dict, List, Optional
from game.config import GameConfig
from game.models import Cell, Coord, BOARD_SIZE, Worker

class Board:
    def __init__(self, game_config: GameConfig, workers: Optional[List[Worker]] = None) -> None:
        self.grid: Dict[Coord, Cell] = {
            (r, c): Cell() for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
        }
        self.workers: List[Worker] = workers or []
        self.game_config = game_config
        self.current_player_index: int = 0
        self.remaining_players: List[str] = list(self.game_config.get_player_ids())
        self.turn_order = list(self.game_config.get_player_ids())
        self.active_players = self.turn_order.copy()

    @property
    def current_player(self) -> str:
        return self.active_players[self.current_player_index]

    def next_turn(self):
        if not self.active_players:
            return  # No one left

        # Convert current_player_index to index if it somehow became a string
        if isinstance(self.current_player_index, str):
            try:
                idx = self.active_players.index(self.current_player_index)
            except ValueError:
                # current player no longer active, start from 0
                idx = -1
        else:
            idx = self.current_player_index

        # Move to next active player
        idx = (idx + 1) % len(self.active_players)
        self.current_player_index = idx

    def in_bounds(self, pos: Coord) -> bool:
        """Return True if pos on the board."""
        row, col = pos
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def get_cell(self, pos: Coord) -> Cell:
        """Return the Cell at pos; raises if out  bounds."""
        if not self.in_bounds(pos):
            raise ValueError("Position out of bounds")
        return self.grid[pos]

    def get_worker(self, worker_id: str) -> Worker:
        """Return the worker with the given id; raises if not found."""
        for worker in self.workers:
            if worker.id == worker_id:
                return worker
        raise ValueError(f"Worker {worker_id} not found")

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

    def eliminate_player(self, pid: str):
        if pid not in self.active_players:
            return


        was_current = (self.active_players[self.current_player_index] == pid)
        removed_idx = self.active_players.index(pid)

        self.active_players.remove(pid)

        # Adjust current_player_index
        if not self.active_players:
            self.current_player_index = 0
        elif was_current:
            # If eliminated was current -> move to next
            self.current_player_index = removed_idx % len(self.active_players)
        elif removed_idx < self.current_player_index:
            # eliminated before current -> shift down
            self.current_player_index -= 1
