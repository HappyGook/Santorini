from game.board import get_legal_moves, BoardState

def coords_to_notation(pos):
    row, col = pos
    return f"{chr(97 + row)}{col+1}"   # 'a'..'e' and 1..5

def get_legal_notation(board: BoardState, player: str):
    moves = get_legal_moves(board, player)
    notation = []
    for worker_pos, new_pos, build_pos in moves:
        worker_str = coords_to_notation(worker_pos)
        new_str = coords_to_notation(new_pos)
        build_str = coords_to_notation(build_pos)
        notation.append(f"{worker_str}-{new_str},{build_str}")
    return notation