from typing import Dict, List
from santorini.models import Cell, Coord, BOARD_SIZE, DOME_LEVEL

class Board:
    def __init__(self) -> None:
        #  generating a 5x5 grid
        self.grid: Dict[Coord, Cell] = {
            (r, c): Cell() for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
        }

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


    def place_worker(self, worker_id: str, pos: Coord) -> None:
        """Put worker on an empty cell"""
        if not self.in_bounds(pos):
            raise ValueError("Position out of bounds")
        if self.is_occupied(pos):
            raise ValueError("Cell is already occupied")
        self.grid[pos].worker_id = worker_id

    def remove_worker(self, pos: Coord) -> None:
        """Clear any worker from a cell """
        if not self.in_bounds(pos):
            raise ValueError("Position out of bounds")
        self.grid[pos].worker_id = None

    def move_worker(self, src: Coord, dst: Coord) -> None:
        """Move worker from src to dst"""
        if not (self.in_bounds(src) and self.in_bounds(dst)):
            raise ValueError("Destination out of bounds")
        wid = self.grid[src].worker_id
        if wid is None: #worker must at src
            raise ValueError("No worker at src")
        if self.is_occupied(dst): #check dst
            raise ValueError("Destination cell is already occupied")
        self.grid[src].worker_id = None
        self.grid[dst].worker_id = wid

    def build_at(self, pos: Coord) -> None:
        """Increase height at pos by 1; 4 becomes a dome and cannot increase further."""
        if not self.in_bounds(pos):
            raise ValueError("Position out of bounds")
        cell = self.grid[pos]
        if cell.height >= DOME_LEVEL:
            raise ValueError("Cannot build above dome level")
        cell.height += 1

    # ---------- optional: debug renderer ----------
    def as_lines(self, cell_width: int = 5) -> list[str]:
  
        lines: list[str] = []

     # width of the left row label like "0: "
        row_label_w = len(str(BOARD_SIZE - 1)) + 2  

        # ---- Header (center each column index within the cell width) ----
        header = " " * row_label_w + " ".join(f"{c:^{cell_width}}" for c in range(BOARD_SIZE))
        lines.append(header)

    # ---- Rows ----
        for r in range(BOARD_SIZE):
            row_label = f"{r:>{row_label_w-2}}: "  # right-align row number
            row_cells = []
            for c in range(BOARD_SIZE):
                cell = self.get_cell((r, c))
            # 'D' for domes (>=4), else numeric height
                h = "D" if cell.height >= DOME_LEVEL else str(cell.height)
                wid = (cell.worker_id or "")
                token = (h + wid)
           
                token = token[:cell_width].ljust(cell_width)
                row_cells.append(token)
            lines.append(row_label + " ".join(row_cells))
        return lines
        