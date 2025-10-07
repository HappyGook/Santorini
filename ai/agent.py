import random

from game.rules import legal_moves, legal_builds
from game.board import BOARD_SIZE
from ai.minimax import minimax

class Agent:
    def __init__(self, player_id: str):
        self.player_id = player_id
        self.phrases=[...]

    def decide_action(self, board_state):
        player_index = board_state.game_config.get_player_index(self.player_id)
        _, action = minimax(board_state, depth=3, player_index=player_index, max_player_index=player_index)
        if action is None:
            return None, None, None
        return action

    def decide_setup(self, board_state):
        empty_cells = [
            (x, y)
            for x in range(BOARD_SIZE)
            for y in range(BOARD_SIZE)
            if board_state.get_cell((x, y)).worker_id is None
        ]
        return random.choice(empty_cells)

    def setup_workers(self, board) -> list[tuple[int, int]]:
        empties = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
               if board.grid[(r, c)].worker_id is None]
        return empties[:2]
