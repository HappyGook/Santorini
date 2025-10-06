from __future__ import annotations
from typing import Tuple, Dict, Literal
from game.board import Board
from game.moves import move_worker, build_block
from game.rules import legal_moves, legal_builds


ActorType = Literal["HUMAN", "AI"]
Coord = Tuple[int, int]

class GameController:
    def __init__(self,board:Board, players: Dict[str,Dict]):
        self.board = board
        self.players = players
        for pid, cfg in self.players.items():
            if cfg["type"] =="AI" and cfg.get("agent") is None:
                raise ValueError(f"Player {pid} is AI but no agent")

    def is_ai_turn(self) -> bool:
        return self.players[self.board.current_player]["type"] == "AI"

    def end_turn(self):
        self.board.current_player = "P2" if self.board.current_player == "P1" else "P1"
        print(f"[DEBUG] end-turn called, switched to {self.board.current_player}")

    def legal_moves_for(self, worker):
        return legal_moves(self.board, worker.pos)

    def legal_builds_for(self, worker):
        return legal_builds(self.board, worker.pos)

    #from game.moves
    def apply_move(self, worker,dst:Tuple[int,int]) -> bool:
        if dst not in self.legal_moves_for(worker):
            return False
        won = move_worker(self.board, worker, dst)
        return (True,won)

        move_worker(self.board, worker, dst)
        return True

    def apply_build(self, worker, build_pos:Tuple[int,int]) -> bool:
        if build_pos not in self.legal_builds_for(worker):
            return False
        build_block(self.board, worker, build_pos)
        return True

#entry point ai to run on tkinter
    def run_ai_turn(self):
        if not self.is_ai_turn():
            return("skip",None,None)

        agent = self.players[self.board.current_player].get("agent")
        worker, move_dst = agent.make_move(self.board)
        if worker is None or move_dst is None:
                  #no legal moves -> opponent wins
            self.end_turn()
            return ("no_moves",None,None)

        self.apply_move(worker, move_dst)
        build_pos = agent.build(worker, self.board)
        if build_pos is not None:
            self.apply_build(worker, build_pos)

        self.end_turn()
        return ("moved", worker, (move_dst, build_pos))