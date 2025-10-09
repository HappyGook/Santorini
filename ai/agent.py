import random

from game.board import BOARD_SIZE
from ai.minimax import minimax, TT, SearchStats

from ai.minimax import minimax
from ai.phrases import PHRASES_BY_PLAYER

class Agent:
    def __init__(self, player_id: str, depth: int =3):
        self.player_id = player_id
        self.depth = depth
        self.phrases=[...]
        self.rng = random.Random()

    def decide_action(self, board_state):

        TT.clear()
        stats = SearchStats()

        player_index = board_state.game_config.get_player_index(self.player_id)
        score, action = minimax(
            board_state,
            depth=self.depth,  # or a fixed int (e.g., 3)
            player_index=player_index,
            max_player_index=player_index,  # maximizing for me
            stats=stats,
            maximizing=True)
        if action is None:
            return None, None, None
        phrase = self.comment_on_eval(max(-1000, min(1000, score))) # normalized because of infinity

        print(f"[AI] depth={self.depth} nodes={stats.nodes} tt_hits={stats.tt_hits} score={score}")
        return action, phrase  # (worker, move, build)

    def decide_setup(self, board_state):
        empty_cells = [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if board_state.get_cell((r, c)).worker_id is None
        ]
        return self.rng.choice(empty_cells)

    def setup_workers(self, board):
        empties = [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if board.grid[(r, c)].worker_id is None
        ]
        return self.rng.sample(empties, 2)

    def comment_on_eval(self, eval_score: float) -> str:
        closest_key = min(PHRASES_BY_PLAYER[self.player_id].keys(), key=lambda x: abs(x - eval_score))
        return PHRASES_BY_PLAYER[self.player_id][closest_key]
