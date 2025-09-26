from typing import Optional
from datetime import datetime
from game.models import Worker, Coord

def coords_to_notation(pos):
    row, col = pos
    return f"{chr(97 + col)}{row+1}"   # 'a'..'e' and 1..5

def notation_to_coords(pos_str):
    col_str, row_str = pos_str
    return  ord(col_str) - ord('a'), int(row_str) - 1

class GameNotation:
    def __init__(self):
        self.moves = []

    def record_setup(self, worker: Worker):
        pos_str = coords_to_notation(worker.pos)
        self.moves.append(f"setup {worker.owner} {worker.id} {pos_str}")

    def record_turn(self, worker: Worker, move_to: Coord, build_at: Coord):
        worker_str = coords_to_notation(worker.pos)
        move_str = coords_to_notation(move_to)
        build_str = coords_to_notation(build_at)
        self.moves.append(f"{worker_str}-{move_str},{build_str}")

    def get_notation(self):
        return "\n".join(self.moves)

    def save(self, filename: Optional[str] = None):
        if filename is None:
            filename = datetime.now().strftime("game%d.%m.%Y.san")
        with open(filename, "w") as f:
            f.write(self.get_notation())

    def clear(self):
        self.moves.clear()