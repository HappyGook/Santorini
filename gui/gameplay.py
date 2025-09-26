from game.board import Board
from game.models import Worker
from game.moves import move_worker, build_block
from game.rules import legal_moves, legal_builds, player_has_moves
from notation import GameNotation, notation_to_coords

def setup_workers(board: Board, notation: GameNotation):
    """Prompt both players to place their workers."""
    players = ["P1", "P2"]
    workers = []

    for player in players:
        for label in ["A", "B"]:
            while True:
                pos_str = input(f"{player}, place your worker {label} (e.g., a1): ")
                try:
                    pos = notation_to_coords(pos_str)
                    cell = board.get_cell(pos)
                    if cell.worker_id is not None:
                        print("Cell already occupied, choose another.")
                        continue
                    worker = Worker(id=f"{player}{label}", owner=player, pos=pos)
                    board.workers.append(worker)
                    cell.worker_id = worker.id
                    workers.append(worker)
                    notation.record_setup(worker)
                    break
                except Exception as e:
                    print("Invalid input:", e)
    return workers

def play_turn(board: Board, notation: GameNotation):
    """Run one turn for the current player."""
    player = board.current_player
    workers = [w for w in board.workers if w.owner == player]

    # Phase 1: move
    while True:
        print(f"\n{player}, choose a worker to move.")
        for i, w in enumerate(workers):
            moves = legal_moves(board, w.pos)
            print(f"{i+1}: {w.id} at {w.pos} can move to {[m for m in moves]}")
        choice = input("Select worker number: ")
        try:
            w = workers[int(choice)-1]
        except:
            print("Invalid worker choice.")
            continue

        moves = legal_moves(board, w.pos)
        if not moves:
            print("This worker has no legal moves.")
            continue

        move_str = input(f"Enter move destination (e.g., b2): ")
        try:
            dst = notation_to_coords(move_str)
        except:
            print("Invalid input format.")
            continue

        if dst not in moves:
            print("Illegal move, try again.")
            continue

        won = move_worker(board, w, dst)
        if won:
            print(f"{player} wins by moving {w.id} to {dst}!")
            return True, w, dst
        break

    # Phase 2: build
    while True:
        builds = legal_builds(board, w.pos)
        if not builds:
            print("No legal build positions. Game over.")
            return True, w, dst

        build_str = input(f"Enter build destination (e.g., c3): ")
        try:
            bpos = notation_to_coords(build_str)
        except:
            print("Invalid input format.")
            continue

        if bpos not in builds:
            print("Illegal build, try again.")
            continue

        build_block(board, w, bpos)
        notation.record_turn(w, dst, bpos)
        break

    if not player_has_moves(board, board.current_player):
        print(f"{board.current_player} has no moves left! They lose.")
        return True, w, dst

    return False, w, dst

def main():
    board = Board([])
    notation = GameNotation()

    print("Welcome to Santorini CLI!")
    setup_workers(board, notation)

    game_over = False
    while not game_over:
        game_over, worker, dst = play_turn(board, notation)

    # Save game
    notation.save()
    print("Game saved.")

if __name__ == "__main__":
    main()