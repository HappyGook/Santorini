from typing import List, Tuple, Dict, Any

from game.board import Board
from game.models import Coord, Worker
from game.rules import all_legal_actions
from ai.heuristics import evaluate_action
import ai.minimax as mm
import ai.maxn as mx
import ai.mcts as mc

Action = Tuple[Worker, Coord, Coord] # (worker, move_to, build_at)

class FallbackStats:  # fallback for search stats
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


def move_ranking(action_score: float, score_breakdown: Dict[str, float], algo: str) -> str:
    """
    Convert heuristic score into logs.
    """
    if algo == "ml":
        if action_score > 0.7:
            return f"Strong move (score: {action_score:.3f}) - model confident"
        elif action_score > 0.3:
            return f"Good move (score: {action_score:.3f}) - model positive"
        elif action_score > -0.3:
            return f"Neutral move (score: {action_score:.3f}) - model uncertain"
        elif action_score > -0.7:
            return f"Weak move (score: {action_score:.3f}) - model negative"
        else:
            return f"Poor move (score: {action_score:.3f}) - model very negative"

    # For heuristic algorithms
    strengths = []
    weaknesses = []

    if score_breakdown.get("height_advantage", 0) > 10:
        strengths.append("strong height advantage")
    elif score_breakdown.get("height_advantage", 0) < -10:
        weaknesses.append("height disadvantage")

    if score_breakdown.get("mobility", 0) > 5:
        strengths.append("excellent mobility")
    elif score_breakdown.get("mobility", 0) < -5:
        weaknesses.append("limited mobility")

    if score_breakdown.get("proximity", 0) > 8:
        strengths.append("great proximity to win")
    elif score_breakdown.get("proximity", 0) < -8:
        weaknesses.append("poor positioning")

    if score_breakdown.get("tower_control", 0) > 6:
        strengths.append("good tower control")
    elif score_breakdown.get("tower_control", 0) < -6:
        weaknesses.append("weak tower control")

    if score_breakdown.get("immediate_win", False):
        return f"WINNING MOVE (score: {action_score:.1f}) - immediate victory!"

    if score_breakdown.get("blocks_opponent_win", False):
        strengths.append("blocks opponent win")

    if score_breakdown.get("opponent_can_win", False):
        weaknesses.append("allows opponent win")

    overall_quality = "excellent" if action_score > 50 else "good" if action_score > 20 else "decent" if action_score > 0 else "poor" if action_score > -30 else "very poor"

    message_parts = [f"Move chosen (score: {action_score:.1f}) - {overall_quality} overall"]

    if strengths:
        message_parts.append(f"due to {', '.join(strengths)}")

    if weaknesses:
        if strengths:
            message_parts.append(f"despite {', '.join(weaknesses)}")
        else:
            message_parts.append(f"with {', '.join(weaknesses)}")

    return " ".join(message_parts)

# return a lsit of action,score on every finished move
def score_all_legal_actions(board:Board,player_id: str) -> List[Tuple[Action, float]]:

    actions_with_scores: List[Tuple[Action, float]] = []
    
    for worker, move,build in all_legal_actions(board,player_id):
        action: Action = (worker,move, build)
        score = float(evaluate_action(board, player_id, action))
        actions_with_scores.append((action, score))
    return actions_with_scores

def normalize_rating(value:float, worst: float, best:float)-> float:

    if best == worst:
        return 0.0 # when only one legal move or no good or bad moves
    
    linear = (value - worst) / (best - worst)

    if linear < 0.0:
        linear = 0.0
    elif linear > 1.0:
        linear = 1.0
    
    return 2.0 * linear -1.0 #map to -1,1]

def label_for_rating(rating:float) -> str:

    if rating <= -0.75:
        return "blunder"
    if rating <= -0.4:
        return "mistake"
    if rating <= -0.15:
        return "inaccuracy"
    if rating <= 0.25:
        return "okay"
    if rating <= 0.75:
        return "good"
    return "excellent"

def rate_move(board_before:Board,player_id: str, action_played: Action)-> Tuple[float,str,dict[str,Any]]:
    # return float in -1,1
    #show player_score, best worst score, best action, num of legal move
    action_scores = score_all_legal_actions(board_before,player_id)
    
    #no legal move
    if not action_scores:
        details:Dict[str,Any] ={
            "player_score": None,
            "best_score": None,
            "worst_score": None,
            "best_actions":[],
            "num_legal_moves": 0,
        }
        return 0.0, "no_moves", details
    
    scores = [score for (_, score) in action_scores]
    best_score = max(scores)
    worst_score = min(scores)

    played_score = None
    for action, score in action_scores:
        if action == action_played:
            played_score = score
            break

        #in case rust try to do some legalmove again

    if played_score is None:
        played_score = worst_score

    rating = normalize_rating(played_score, worst_score, best_score)
    label = label_for_rating(rating)

    best_actions = [action for (action, score) in action_scores if score == best_score]

    details: Dict[str, Any] = {
        "played_score": played_score,
        "best_score": best_score,
        "worst_score": worst_score,
        "best_actions": best_actions,
        "num_legal_moves": len(action_scores),
    }

    return rating, label, details