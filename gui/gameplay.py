from __future__ import annotations
from typing import Tuple, List, Optional, Dict
from game.board import Board
from game.models import Worker, Cell, BOARD_SIZE
from game.moves import move_worker, build_block
from game.rules import legal_moves, legal_builds, player_has_moves
from gui.notation import GameNotation, notation_to_coords, coords_to_notation
from ai.agent import Agent



Coord = Tuple[int, int]

def render_board(b: Board) -> None: #print board and height and worker id

    letters = [chr(ord('a') + c) for c in range(BOARD_SIZE)]
    print("\n   " + "  ".join(letters))
    for r in range(BOARD_SIZE):
        row_bits = []
        for c in range(BOARD_SIZE):
            cell = b.grid[(r, c)]
            tag = (cell.worker_id or " ").ljust(2)
            row_bits.append(f"{cell.height}{tag}")
        print(f"{r+1} " + " ".join(row_bits))
    print()
   
def player_workers(b: Board, player:str) -> List[Worker]:

    return [w for w in b.workers if w.owner == player] #filter workers by player

def any_moves_for(b: Board, player: str) -> bool:
    return any(legal_moves(b, w.pos) for w in player_workers(b, player))  #check if  worker has any legal moves noch


def input_coord(prompt: str) -> Coord:
    s = input(prompt).strip()   #parse a1 to row,col
    return notation_to_coords(s)

def input_worker_id(prompt: str, choices: List[str]) -> str:    #pick worker id from list
     
    while True:
        s = input(prompt).strip()
        if s in choices:
            return s
        print(f"Invalid choice, must be one of {choices}")

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

def play_turn(board: Board, notation: GameNotation, ai_agents: Optional[Dict[str, Agent]] = None):
    """Run one turn for the current player."""
    player = board.current_player
    workers = [w for w in board.workers if w.owner == player]

    # === AI PATH ===
    if ai_agents and player in ai_agents:
        agent = ai_agents[player]

        # Move phase
        w, dst = agent.make_move(board)
        old = w.pos
        won = move_worker(board, w, dst)
        if won:
            print(f"{player} wins by moving {w.id} to {coords_to_notation(dst)}!")
            notation.record_turn(old, dst, dst)  # record move (build dummy same as move)
            return True, w, dst

        # Build phase
        bpos = agent.build(board)
        build_block(board, w, bpos)
        notation.record_turn(old, dst, bpos)

        # Check if opponent has moves
        if not player_has_moves(board, board.current_player):
            print(f"{board.current_player} has no moves left! They lose.")
            return True, w, dst

        # Switch player
        board.current_player = "P2" if player == "P1" else "P1"
        return False, w, dst

    # === HUMAN PATH ===
    # Move phase
    while True:
        print(f"\n{player}, choose a worker to move.")
        for i, w in enumerate(workers):
            moves = legal_moves(board, w.pos)
            moves_notation = [coords_to_notation(m) for m in moves]
            print(f"{i+1}: {w.id} at {coords_to_notation(w.pos)} can move to {moves_notation}")

        choice = input("Select worker (A/B): ").strip().upper()
        if choice not in ["A", "B"]:
            print("Invalid worker choice. Must be A or B.")
            continue
        w = workers[ord(choice) - ord('A')]

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

        old = w.pos
        won = move_worker(board, w, dst)
        if won:
            print(f"{player} wins by moving {w.id} to {coords_to_notation(dst)}!")
            notation.record_turn(old, dst, dst)  # record move (build dummy)
            return True, w, dst
        break

    # Build phase
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
        notation.record_turn(old, dst, bpos)
        break

    # Check if opponent has moves
    if not player_has_moves(board, board.current_player):
        print(f"{board.current_player} has no moves left! They lose.")
        return True, w, dst

    # Switch player
    board.current_player = "P2" if player == "P1" else "P1"
    return False, w, dst