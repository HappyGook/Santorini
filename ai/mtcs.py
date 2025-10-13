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

def mcts_search(root_board, player_index, iterations=500, game_config=None):
    root = MCTSNode(root_board, player_index)
    for i in range(iterations):
        node = select(root)
        child = expand(node, game_config)
        reward = simulate(child.board, player_index)
        backpropagate(child, reward)

    # choose best move by visit count
    best_child = max(root.children.values(), key=lambda c: c.visits)
    # prepare vector-like output (like maxn)
    vector = [0] * len(root.board.players)
    vector[player_index] = best_child.value / (best_child.visits or 1)
    return vector, best_child.action

def select(node):
    while node.children:
        node = max(node.children.values(), key=uct_value)
    return node

def uct_value(node):
    if node.visits == 0:
        return INF
    C = 1.4
    return (node.value / node.visits) + C*((math.log(node.parent.visits) / node.visits) ** 0.5)

def expand(node, game_config=None):
    actions = [a for a in generate_actions(node.board, node.player_index)
               if a not in node.children]
    if not actions:
        return node
    ordered_actions = order_moves(node.board, actions)
    action = ordered_actions[0]  # select the best move according to heuristic
    new_board = node.board.clone()
    worker, move, build = action
    move_worker(new_board, worker, move)
    build_block(new_board, worker, build)
    next_index = next_player_index(node.player_index, len(new_board.players))
    child = MCTSNode(new_board, next_index, parent=node, action=action)
    node.children[action] = child
    return child

def simulate(board, player_index, steps=3):
    temp_board = board.clone()
    current_index = player_index
    for _ in range(steps):
        actions = generate_actions(temp_board, current_index)
        if not actions:
            break
        action = random.choice(actions)
        worker, move, build = action
        move_worker(temp_board, worker, move)
        build_block(temp_board, worker, build)
        current_index = next_player_index(current_index, len(temp_board.players))
    return evaluate(temp_board, player_index)

def backpropagate(node, reward):
    while node:
        node.visits += 1
        node.value += reward
        node = node.parent

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
