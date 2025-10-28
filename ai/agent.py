from __future__ import annotations

import random
from typing import Optional, Tuple, Literal
import torch
from sympy.categories import Object
from ml.encode import encode_board, encode_action
from game.rules import all_legal_actions

import ai.minimax as mm
import ai.maxn as mx
import ai.mcts as mc

from game.board import BOARD_SIZE
from ai.phrases import PHRASES_BY_PLAYER


class FallbackStats: # fallback for search stats 
    __slots__ = ("nodes", "tt_hits")
    def __init__(self) -> None:
        self.nodes = 0
        self.tt_hits = 0
    def bump(self) -> None:
        self.nodes += 1

def make_stats():
    # Prefer an algos for SearchStats 
    Stats = (getattr(mm, "SearchStats", None) or getattr(mx, "SearchStats", None) or getattr(mc, "SearchStats", None))
    if Stats is None:
        return FallbackStats()
    return Stats()



AlgoName = Literal["minimax", "maxn", "mcts","ml"]

class Agent:
    def __init__(self, player_id: str, algo: AlgoName = "minimax",
                 depth: int =3, iters: Optional[int] = None,
                 rng_seed: Optional[int] = None, model: Optional[Object]=None):
        self.player_id = player_id
        self.depth = depth
        self.rng = random.Random(rng_seed)
        self.algo: AlgoName = algo
        self.model = model
        self.iters = iters

    def decide_action(self, board_state)-> Tuple[float | list[float], Optional[tuple]]:

        game_config = board_state.game_config
        player_index = game_config.get_player_index(self.player_id)

        # New stats per decision and clear per-algo TT
        stats = make_stats()
        if hasattr(mm, "TT"): mm.TT.clear()
        if hasattr(mx, "TT"): mx.TT.clear()
        if hasattr(mc, "TT"): mc.TT.clear()

        if self.algo == "minimax":
           
            score, action = mm.minimax(
                board_state,
                depth=self.depth,
                player_index=player_index,
                max_player_index=player_index,  # maximize 
                stats=stats,
                maximizing=True
            )
            eval_value = score

        elif self.algo == "maxn":
            vector, action = mx.maxn(
                board_state,
                depth=self.depth,
                player_index=player_index,
                game_config=game_config,
                stats=stats
            )
            eval_value = vector

        elif self.algo == "ml":
            if self.model is None:
                raise Exception("A trained model must be passed for 'ml' mode.")
            legal_actions = all_legal_actions(board_state, player_index)
            print(f"[DEBUG] Player {self.player_id} legal_actions={legal_actions}")
            if not legal_actions:
                raise Exception(f"No legal actions for player {self.player_id} at this board state")
            board_tensor = encode_board(board_state, player_index)
            actions_tensor = torch.stack([encode_action(board_state, action, player_index) for action in legal_actions])
            values = self.model.evaluate_actions(board_tensor, actions_tensor)
            best_index = torch.argmax(values).item()
            action = legal_actions[best_index]
            eval_value = values[best_index].item()

        else:  # "mcts"
            # Allow overriding iterations
            vector, action = mc.mcts(
                board_state,
                player_index=player_index,
                game_config=game_config,
                iters=self.iters,
                depth=self.depth,
                stats=stats
            )
            eval_value = vector

        print(f"[{self.player_id}][{self.algo}] nodes={stats.nodes} tt_hits={stats.tt_hits}")

        return eval_value, action
        

    def decide_setup(self, board_state):
        empty_cells = [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if board_state.grid[(r, c)].worker_id is None
        ]
        return self.rng.choice(empty_cells) if empty_cells else None

    def setup_workers(self, board):
        empties = [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if board.grid[(r, c)].worker_id is None
        ]
        return self.rng.sample(empties, 2)

    def comment_on_eval(self, eval_score: float) -> str:
        mapping = PHRASES_BY_PLAYER.get(self.player_id)
        if not mapping:
            return ""
        closest_key = min(mapping.keys(), key=lambda x: abs(x - float(eval_score)))
        return mapping[closest_key]
