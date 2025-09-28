from game.board import Board
from gui.notation import GameNotation
from gui.gameplay import setup_workers, play_turn, render_board
from game.board import Board

def main():
    print("Welcome to Santorini CLI!")
    board = Board()
    notation = GameNotation()
    setup_workers(board, notation)
    render_board(board)
    if not getattr(board, 'current_player', None):
        board.current_player = "P1"  # Default starting player if not set
    while True:
        if play_turn(board, notation):
            break
    try:
        path = notation.save()
        print(f"Game saved to {path}")
    
    except Exception:
        pass
if __name__ == "__main__":
    main()

    