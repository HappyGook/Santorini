import math
import random
from typing import Tuple, List
from ai.heuristics import evaluate
from game.moves import move_worker, build_block, find_worker_by_id
from game.rules import legal_moves, legal_builds

from ml.inference import ml_inference
"""
    MCTS
    1 selection - follow tree with puct to leaf
    2 expansion - expand one new child
    3 simulation - rollout with heuristic or play randomly
    4 backpropagation - update values along the path to root

"""

# the type for ai's action (worker, move, build)
Action = Tuple[object, Tuple[int, int], Tuple[int, int]]

TT = {}

def get_node(board, player_index, **kwargs):
    key = (hash(board), player_index)
    if key in TT:
        return TT[key]
    n = MCTSNode(board, player_index, **kwargs)
    TT[key] = n
    return n


class MCTSNode:
    def __init__(self, board, player_index, parent=None, action=None, prior = 0.0):
        self.board = board
        self.player_index = player_index
        self.parent = parent
        self.action = action    # action that led from parent -> this node
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0    #W total accumulated value from rollouts
        self.prior = prior   #  prior probability from heuristic/policy
    @property
    def q (self):       #q (mean value) = W / N
        return 0.0 if self.visits ==0 else self.value_sum / self.visits


# PUCT selection formula
# PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
# Q  : mean value of the child (how good so far)
# P  : prior probability (how promising a move seemed before exploring)
# Np : parent visit count
# Ni : child visit count
#stant controlling how much to favor unexplored actions
def puct(child, parent_visits, c_puct=1.25):
    u = c_puct *child.prior * math.sqrt(max(1, parent_visits)) / (1 + child.visits)
    return child.q + u

# Selection: walk down the tree until we reach a leaf (no children yet)
def select(node):
    while node.children:
        node = max(node.children.values(), key=lambda c: puct(c, node.visits))
    return node


#from heuristics that produce prior score for each action
# reward climbing higher, building on top of level-3
def action_prior(board, player_id, action):
    worker_id, move, build = action
    h_move = board.grid[move].height    
    h_build = board.grid[build].height
    # Height difference between target and current position
    climb = max(0, h_move - board.grid[find_worker_by_id(board, worker_id).pos].height)
    #make dome get bonus score
    cap = 1.0 if h_build == 3 else 0.0
     # Weighted linear combination → heuristic "score"
    return 3*climb +5 *cap + h_move 

# Convert a list of raw scores into probabilities (sum = 1)
def softmax(x, temp=1.0):
    m = max(x); e = [math.exp((v-m)/temp) for v in x]; s = sum(e)
    return [v/s for v in e]

# Expansion: add one unexplored child node to the tree
def expand(node, game_config, stats):
    player_id = game_config.get_player_id(node.player_index)
    actions = generate_actions(node.board, player_id)

      # skip if all actions already expanded
    unexplored = [a for a in actions if a not in node.children]
    if not unexplored: 
        return None

    # get heuristic priors P for actions
    scores = [action_prior(node.board, player_id, a) for a in unexplored]
    priors = softmax(scores, temp=1.0)

    # choose one action weighted by priors
    a = random.choices(unexplored, weights=priors, k=1)[0]
    new_board = node.board.clone()
    wid, move, build = a
    w = find_worker_by_id(new_board, wid)

    # Apply move and check terminal win
    if move_worker(new_board, w, move):   # reached win state
        child = MCTSNode(new_board, node.player_index, parent=node, action=a, prior=priors[unexplored.index(a)])
        child.visits = 1
        child.value_sum = 1.0 #win reward = +1
        node.children[a] = child
        return child
    
    # Normal non-terminal move
    build_block(new_board, w, build)
    next_index = (node.player_index + 1) % len(game_config.player_ids)
    
    
    child = MCTSNode(new_board, next_index, parent=node, action=a, prior=priors[unexplored.index(a)])
    node.children[a] = child
    stats.bump()    # count expanded nodes
    return child

# Backpropagation: update visit counts and total value along the path
def backpropagate(node, reward, root_player):
    while node:
        node.visits += 1
        # perspective: positive if good for root_player
        sign = +1 if node.player_index == root_player else -1
        node.value_sum += sign * reward
        node = node.parent

#main mcts search function
def mcts_search(board, player_index, game_config, stats, iterations=500, ml_model=None, use_nn=False):
    """
       MCTS search that returns the same format as maxn: (vector, action)
    """
    root = MCTSNode(board, player_index)
    ROOT = player_index
    for i in range(iterations):

        node = select(root)#1

        child = expand(node, game_config, stats)#2
        if child is None:
            child = node

        reward = simulate(
        child.board,
        child.player_index,
        game_config,
        stats,
        ml_model if use_nn else None
    )
        
        backpropagate(child, reward, ROOT)#4

    if not root.children:
        # No valid moves found
        return None, None

    # choose best move by visit count
    best_child = max(root.children.values(), key=lambda c: c.visits)

    num_players = len(game_config.player_ids)
    vector = [0.0] * num_players
    if best_child.visits > 0:
        vector[player_index] = best_child.q

    # convert action from (worker_id, move, build) to (worker_obj, move, build)
    if best_child.action is not None:
        worker_id, move, build = best_child.action
        worker_obj = find_worker_by_id(root.board, worker_id)
        action = (worker_obj, move, build)
    else:
        action = None

    return vector, action

def simulate(board, player_index, game_config, stats, ml_model=None, steps=12, eps=0.15):

    """Fast heuristic-based simulation from the current position"""
    temp_board = board.clone()
    current_index = player_index

    for _ in range(steps):
        player_id = game_config.get_player_id(current_index)
        actions = generate_actions(temp_board, player_id)
        if not actions:
            break

        pri = [action_prior(temp_board, player_id, a) for a in actions]      
        a = random.choice(actions) if random.random() < eps else actions[max(range(len(actions)), key=lambda i: pri[i])]
        
        wid, move, build = a
        w = find_worker_by_id(temp_board, wid)
        if move_worker(temp_board, w, move):    # win for cur
                return 1.0 if current_index == player_index else -1.0
        build_block(temp_board, w, build)
        current_index = (current_index + 1) % len(game_config.player_ids)

    value = None

    # NN-based evaluation
    if ml_model is not None:
        value, _ = ml_inference(temp_board, player_index, ml_model, stats)

    # if NN returns no scalar, fall back to heuristic below
    if value is not None:
        value = max(-0.5, min(0.5, float(value)))
        return value
    # fallback heuristic
    root_pid = game_config.get_player_id(player_index)
    v = evaluate(temp_board, root_pid)
    return max(-0.5, min(0.5, v / 100.0))



def mcts(board, depth, player_index, game_config, stats, iters=None, ml_model=None, use_nn=False, **kwargs):
    """
    MCTS wrapper that matches the maxn function signature for easy agent integration
    The depth parameter is converted to iterations
    """
    iterations = int(iters) if iters is not None else max(1000, depth * 400)
    return mcts_search(board, player_index, game_config, stats, iterations, ml_model=ml_model, use_nn=use_nn)

# A replacement for rules.all_legal_actions for mcts
# Req due to hashing
def generate_actions(board, player_id) -> List[Action]:
    actions = []
    workers = [w for w in board.workers if w.owner == player_id]
    for worker in workers:
        for move in legal_moves(board, worker.pos):
            for build in legal_builds(board, move):
                actions.append((worker.id, move, build))
    return actions


class SearchStats:
    def __init__(self):
        self.nodes = 0
        self.tt_hits = 0

    def bump(self): self.nodes += 1