from __future__ import annotations
import math
import random
from typing import Optional, Tuple, Literal
from sympy.categories import Object
import ai.minimax as mm
import ai.maxn as mx
import ai.mcts as mc
from ai.move_rating import move_ranking
from game.board import BOARD_SIZE
from ai.phrases import PHRASES_BY_PLAYER
from ai.mcts import SearchStats
from ai.heuristics import find_win_in_one, detailed_eval
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



AlgoName = Literal["minimax", "maxn", "mcts","rust_mcts","mcts_NN"]

class Agent:
    def __init__(self, player_id: str, algo: AlgoName = "minimax",
                 depth: int =3, iters: Optional[int] = None,
                 rng_seed: Optional[int] = None, model: Optional[Object]=None):
        self.player_id = player_id
        self.depth = depth
        self.rng = random.Random(rng_seed)
        self.algo: AlgoName = algo
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

        elif self.algo == "minimax_NN":

            score, action = mm.minimax(
                board_state,
                depth=self.depth,
                player_index=player_index,
                max_player_index=player_index,  # maximize
                stats=stats,
                ml_model=self.model,
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

            game_config = board_state.game_config
            player_index = game_config.get_player_index(self.player_id)
            num_players = game_config.num_players

            win_action = find_win_in_one(board_state, self.player_id)
            if win_action is not None:
            # Immediate win – no need to call MCTS
                eval_value = 1.0
                action = win_action
                return eval_value, action

            iterations = self.iters if self.iters is not None else 400

            value, best_action = rust.run_mcts_python_rules(
            board_state,
            player_index=player_index,
            iterations=iterations,
            num_players=num_players,
        )

            eval_value = value
        
            if best_action is None:
                action = None
            else:
                worker_id, move_pos, build_pos = best_action  # from Rust: id string + coords
                # find the real Worker object on the current board
                worker = next(w for w in board_state.workers if w.id == worker_id)
                action = (worker, move_pos, build_pos)


        elif self.algo == "ml":
            from ml.inference import ml_inference
            if self.model is None:
                raise Exception("A trained model must be passed for ML mode")
            eval_value, action = ml_inference(board_state, player_index, self.model, stats)

        elif self.algo == "maxn_NN":
            vector, action = mx.maxn(
                board_state,
                depth=self.depth,
                player_index=player_index,
                game_config=game_config,
                stats=stats,
                ml_model=self.model
            )
            eval_value = vector

        else:
            
            raise ValueError(f"Unknown algorithm: {self.algo}")

        print(
            f"[{self.player_id}][{self.algo}] nodes={stats.nodes} tt_hits={stats.tt_hits}"
        )

        if action is not None:
            try:
                # Get detailed evaluation of the chosen action
                score_breakdown = detailed_eval(board_state, self.player_id, action)

                # Generate human-friendly log message
                if self.algo == "ml":
                    log_message = move_ranking(float(eval_value), {}, "ml")
                else:
                    # For heuristic algorithms, use the score breakdown
                    total_score = sum(score_breakdown.values()) if isinstance(score_breakdown, dict) else float(
                        eval_value)
                    log_message = move_ranking(total_score, score_breakdown, self.algo)

                print(f"[{self.player_id}] {log_message}")

            except Exception as e:
                print(f"[{self.player_id}] Move evaluation failed: {e}")

        # ALWAYS return a (value, action) tuple
        return eval_value, action


    def setup_workers(self, board_state):
        empty_cells = [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if board_state.grid[(r, c)].worker_id is None
        ]

        # worker 1 picks free-est spot
        free_scores=[]
        for cell in empty_cells:
            free = 0
            for nb_cell in board_state.neighbors(cell):
                if not board_state.is_occupied(nb_cell):
                    free += 1
            free_scores.append(free)

        # exponential semi-randomness
        weights1 = [math.exp(score) for score in free_scores]
        worker1 = random.choices(empty_cells, weights=weights1, k=1)[0]

        # worker 2 tries to get the best next to worker 1
        def distance(a,b):
            return abs(a[0]-b[0])+abs(a[1]-b[1])

        weights2 = []
        for cell in empty_cells:
            if cell == worker1:
                weights2.append(0)
            else:
                d = distance(cell, worker1)
                # closer is heavier
                weights2.append(math.exp(-d))

        worker2 = random.choices(empty_cells, weights=weights2, k=1)[0]

        return worker1, worker2

    def comment_on_eval(self, eval_score: float) -> str:
        mapping = PHRASES_BY_PLAYER.get(self.player_id)
        if not mapping:
            return ""
        closest_key = min(mapping.keys(), key=lambda x: abs(x - float(eval_score)))
        return mapping[closest_key]
