from typing import Dict, List
from models import Cell, Coord, BOARD_SIZE, DOME_LEVEL

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


def get_legal_moves(board, player):
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    legal_moves = []
    for worker_pos in board.workers[player]:
        row, column = worker_pos
        current_height = board.board[row][column][0]

        for dir_row, dir_column in directions:
            pos_row, pos_column = row + dir_row, column + dir_column

            # check inside board
            if 0 <= pos_row <=4 and 0 <= pos_column <=4:
                if board.board[pos_row][pos_column][1] is None:
                    if current_height>=board.board[pos_row][pos_column][0]-1:
                        for direction_build_row, direction_build_column in directions:
                            build_row, build_column = pos_row + direction_build_row, pos_column + direction_build_column
                            if 0 <= build_row <=4 and 0 <= build_column <=4:
                                if board.board[build_row][build_column][1] is None and board.board[build_row][build_column][0]<=3:
                                    legal_moves.append((worker_pos, (pos_row, pos_column), (build_row,build_column)))

    return legal_moves
print(get_legal_moves(board, "A"))


def make_move(board, player, move):
    legal_moves = get_legal_moves(board, player)
    if len(legal_moves) == 0:
        raise Exception("No legal moves") # Means player lost, needs handling
    if move not in legal_moves:
        raise Exception("Illegal move")

    worker_pos, new_pos, build_pos = move

    # Clear old worker position
    board.board[worker_pos[0]][worker_pos[1]] = (board.board[worker_pos[0]][worker_pos[1]][0], None)

    # Move worker to new position
    old_height, _ = board.board[new_pos[0]][new_pos[1]]
    board.board[new_pos[0]][new_pos[1]] = (old_height, player)

    # Update worker tracking
    for i, w in enumerate(board.workers[player]):
        if w == worker_pos:
            board.workers[player][i] = new_pos
            break

    # --- Check win: moving onto level 3 ---
    if old_height == 3:
        return True  # Player just won

    # Apply build (increase height, leave occupant unchanged)
    build_height, occ = board.board[build_pos[0]][build_pos[1]]
    board.board[build_pos[0]][build_pos[1]] = (build_height + 1, occ)  # ???
    board.current_player = "B" if board.current_player == "A" else "A" # Changes current player, for a loop

    return False  # normal continue

