import math
import random
from typing import Tuple, List
from ai.heuristics import evaluate
from game.config import GameConfig
from game.moves import move_worker, build_block
from game.rules import legal_moves, legal_builds
from functools import lru_cache

# infinity const for evaluation win/lose
INF = 10 ** 9
# the type for ai's action (worker, move, build)
Action = Tuple[object, Tuple[int, int], Tuple[int, int]]

TT = {}

class MCTSNode:
    def __init__(self, board, player_index, parent=None, action=None):
        self.board = board
        self.player_index = player_index
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0

def mcts_search(board, player_index, game_config, stats, iterations=500):
    """
       MCTS search that returns the same format as maxn: (vector, action)
    """
    root = MCTSNode(board, player_index)
    for i in range(iterations):
        node = select(root)
        child = expand(node, game_config, stats)
        if child is None:
            child = node
        reward = simulate(child.board, child.player_index, game_config)
        backpropagate(child, reward)

    if not root.children:
        # No valid moves found
        return None, None


    # choose best move by visit count
    best_child = max(root.children.values(), key=lambda c: c.visits)

    num_players = len(game_config.player_ids)
    vector = [0.0] * num_players
    if best_child.visits > 0:
        vector[player_index] = best_child.value / best_child.visits

    return vector, best_child.action

def select(node):
    while node.children:
        node = max(node.children.values(), key=uct_value)
    return node

def uct_value(node):
    """Upper Confidence Bound for Trees"""
    if node.visits == 0:
        return INF

    C = 1.4  # exploration parameter
    exploitation = node.value / node.visits
    exploration = C * math.sqrt(math.log(node.parent.visits) / node.visits)
    return exploitation + exploration


def expand(node, game_config, stats):
    """Expand a node by adding one child"""
    player_id = game_config.get_player_id(node.player_index)
    actions = generate_actions(node.board, player_id)

    # Filter out actions that are already children
    unexplored_actions = [a for a in actions if a not in node.children]

    if not unexplored_actions:
        return None  # fully expanded

    ordered_actions = order_moves(node.board, unexplored_actions)
    action = ordered_actions[0]  # select the best move according to heuristic

    worker, move, build = action
    new_board = node.board.clone()

    new_worker = find_worker_by_id(new_board, worker.id)
    if new_worker is None:
        return None

    won = move_worker(new_board, worker, move)
    if won:
        num_players = len(game_config.player_ids)
        winning_vector = [-INF] * num_players
        winning_vector[node.player_index] = INF
        child = MCTSNode(new_board, node.player_index, parent=node, action=action)
        child.value = INF
        child.visits = 1
        node.children[action] = child
        return child

    build_block(new_board, worker, build)

    next_index = (node.player_index + 1) % len(game_config.player_ids)
    child = MCTSNode(new_board, next_index, parent=node, action=action)
    node.children[action] = child
    stats.bump()
    return child

def simulate(board, player_index, game_config, steps=3):
    """Random simulation from the current position"""
    temp_board = board.clone()
    current_index = player_index
    for _ in range(steps):
        player_id = game_config.get_player_id(current_index)
        actions = generate_actions(temp_board, current_index)
        if not actions:
            break

        action = random.choice(actions)
        worker, move, build = action

        temp_worker = find_worker_by_id(temp_board, worker.id)
        if temp_worker is None:
            break
        won = move_worker(temp_board, worker, move)
        if won:
            return INF if current_index == player_index else -INF

        build_block(temp_board, worker, build)
        current_index = (current_index + 1) % len(game_config.player_ids)

    player_id = game_config.get_player_id(player_index)
    return evaluate(temp_board, player_id)

def backpropagate(node, reward):
    while node:
        node.visits += 1
        node.value += reward
        node = node.parent

def mcts(board, depth, player_index, game_config, stats, **kwargs):
    """
    MCTS wrapper that matches the maxn function signature for easy agent integration
    The depth parameter is converted to iterations (depth * 100)
    """
    iterations = max(100, depth * 100)  # Scale depth to reasonable iteration count
    return mcts_search(board, player_index, game_config, stats, iterations)


def generate_actions(board, player_index) -> List[Action]:
    actions = []
    workers = [w for w in board.workers if w.owner == player_index]
    for worker in workers:
        for move in legal_moves(board, worker.pos):
            for build in legal_builds(board, move):
                actions.append((worker, move, build))
    return actions


def next_player_index(current_index: int, num_players: int) -> int:
    """Simple modular rotation - can be moved to config"""
    return (current_index + 1) % num_players


def next_player(player_id: str, game_config: GameConfig) -> str:
    current_index = game_config.get_player_index(player_id)
    next_index = game_config.next_player_index(current_index)
    return game_config.get_player_id(next_index)


def find_worker_by_id(board, worker_id):
    return next((w for w in board.workers if w.id == worker_id), None)


class SearchStats:
    def __init__(self):
        self.nodes = 0
        self.tt_hits = 0

    def bump(self): self.nodes += 1


def order_moves(board, moves):
    # moves: list[(worker, move_to, build_to)]
    def score_move(m):
        (w, mv, bd) = m
        r0, c0 = w.pos;
        r1, c1 = mv
        h0 = board.grid[(r0, c0)].height;
        h1 = board.grid[(r1, c1)].height
        delta = h1 - h0  # climbing is good in Santorini
        # prefer building next to our worker and capping enemy’s towers later
        return (delta, h1, -(board.grid[bd].height))

    return sorted(moves, key=score_move, reverse=True)
