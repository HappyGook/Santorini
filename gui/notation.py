from typing import Optional, Tuple
from datetime import datetime
from game.models import Worker, Coord

def coords_to_notation(rc: Tuple[int, int]) -> str:
    row, col = rc
    return chr(ord('a') + col) + str(row + 1)


def notation_to_coords(s: str) -> Tuple[int, int]:
    s = s.strip().lower()
    if len(s) < 2 or not s[0].isalpha() or not s[1:].isdigit():
        raise ValueError(f"Bad coordinate notation: {s}")

    col = ord(s[0]) - ord('a')   # letters to col a->0 b->1 c->2 d->3 e->4
    row = int(s[1:]) - 1         # num map to rows
    return (row, col)            # fix cos u return (col, row last time)

class GameNotation:
    def __init__(self):
        self.moves = []

    def record_setup(self, worker: Worker):
        pos_str = coords_to_notation(worker.pos)
        self.moves.append(pos_str)

    def record_turn(self, old_pos: Coord, move_to: Coord, build_at: Optional[Coord] = None):
        old_str = coords_to_notation(old_pos)
        move_str = coords_to_notation(move_to)
        if build_at is not None:
            build_str = coords_to_notation(build_at)
            self.moves.append(f"{old_str}-{move_str},{build_str}")
        else:
            # For winning moves where no build occurs
            self.moves.append(f"{old_str}-{move_str}")

    def get_notation(self):
        return "\n".join(self.moves)

    def save(self, filename: Optional[str] = None):
        if filename is None:
            filename = datetime.now().strftime("game%d.%m.%Y.san")
        with open(filename, "w") as f:
            f.write(self.get_notation())

    def clear(self):
        self.moves.clear()