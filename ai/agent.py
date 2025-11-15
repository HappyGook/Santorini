from __future__ import annotations

import random
from typing import Optional, Tuple, Literal
from sympy.categories import Object

import ai.minimax as mm
import ai.maxn as mx
import ai.mcts as mc

from game.board import BOARD_SIZE
from ai.phrases import PHRASES_BY_PLAYER
from ai.mcts import mcts, SearchStats


import rust

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



AlgoName = Literal["minimax", "maxn", "mcts","rust_mcts""mcts_NN"]

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
        self.model = model

    def decide_action(self, board_state)-> Tuple[float | list[float], Optional[tuple]]:
        stats = SearchStats()
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

        elif self.algo == "mcts":
            print(f"[{self.player_id}] using Python MCTS ...")
            iterations = int(self.iters) if self.iters is not None else max(1000, self.depth * 400)

            vector, action = mc.mcts_search(
                board_state,
                player_index=player_index,
                game_config=game_config,
                stats=stats,
                iterations=iterations,
    )
            eval_value = vector

        elif self.algo == "mcts_NN":
            print(f"[{self.player_id}] using MCTS-NN ...")
            # NEW: MCTS that uses NN at simulation leaf
            vector, action = mc.mcts(
                board_state,
                player_index=player_index,
                game_config=game_config,
                iters=self.iters,
                depth=self.depth,
                stats=stats,
                ml_model=self.model,
                use_nn=True,
            )
            eval_value = vector


        elif self.algo == "rust_mcts":
    
            print(f"[{self.player_id}] using Rust hybrid MCTS ...")

            value, best_action = rust.run_mcts_python_rules(
                board_state,
                player_index=player_index,
                iterations=self.iters or 500,
                num_players=board_state.game_config.num_players,
            )

            if best_action is None:
                action = None
            else:
                wid, move, build = best_action
                worker = next((w for w in board_state.workers if w.id == wid), None)
                action = (worker, move, build) if worker else None
            eval_value = value


        elif self.algo == "ml":
            from ml.inference import ml_inference
            if self.model is None:
                raise Exception("A trained model must be passed for ML mode")
            eval_value, action = ml_inference(board_state, player_index, self.model, stats)

        else:
            
            raise ValueError(f"Unknown algorithm: {self.algo}")

        print(
            f"[{self.player_id}][{self.algo}] nodes={stats.nodes} tt_hits={stats.tt_hits}"
        )

        # ALWAYS return a (value, action) tuple
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
