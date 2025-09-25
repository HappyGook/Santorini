class BoardState:
    def __init__(self):
        # Each cell = (height, occupant)
        # occupant = None, "A", or "B" or a Dome (define?)
        self.board = [[(0, None) for _ in range(5)] for _ in range(5)]
        self.workers = {"A": [(0,0),(4,4)], "B": [(0,0),(0,0)]}
        self.current_player = "A"

board = BoardState()

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

