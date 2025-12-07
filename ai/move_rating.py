from typing import List, Tuple, Dict, Any

from game.board import Board
from game.models import Coord, Worker
from game.rules import all_legal_actions
from ai.heuristics import evaluate_action

Action = Tuple[Worker, Coord, Coord] # (worker, move_to, build_at)

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

    if rating <= -0.8:
        return "blunder"
    if rating <= -0.5:
        return "mistake"
    if rating <= -0.15:
        return "inaccuracy"
    if rating <= 0.2:
        return "okay"
    if rating <= 0.5:
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

    sorted_scores = sorted(scores)
    n =len(sorted_scores)

    if n %2 ==1:
        median_score = sorted_scores[n//2]
    else:
        median_score = 0.5 * (sorted_scores[n//2 -1] + sorted_scores[n//2])  

    q1_score = sorted_scores[n//4] # 25%
    q3_score = sorted_scores[(3*n)//4] #75% 

    NEAR_BEST_EPS = 1e-6 #almost best still excellent

    if played_score >= best_score - NEAR_BEST_EPS:
        rating = 1.0
    elif played_score >= q3_score:
        rating = 0.5
    elif played_score >= q1_score:
        rating = 0.0
    elif played_score > worst_score:
        
        mid_low = 0.5 * (worst_score + q1_score)
        if played_score >= mid_low:
            rating = -0.2
        else:
            rating = -0.5
    else:
        rating = -0.9

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