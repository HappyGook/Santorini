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

    # convert action from (worker_id, move, build) to (worker_obj, move, build)
    if best_child.action is not None:
        worker_id, move, build = best_child.action
        worker_obj = find_worker_by_id(root.board, worker_id)
        action = (worker_obj, move, build)
    else:
        action = None

    return vector, action


def select(node):
    while node.children:
        node = max(node.children.values(), key=lambda c: uct_value(c, node.player_index))
    return node


def uct_value(node, parent_player_index):
    """Upper Confidence Bound for Trees - adversarial version"""
    if node.visits == 0:
        return INF

    C = 1.4  # exploration parameter
    exploitation = node.value / node.visits

    # If opponent's node, we want to minimize their value (adversarial)
    if node.player_index != parent_player_index:
        exploitation = -exploitation

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

    # Random exploration - let MCTS learn through UCT which moves are best
    action = random.choice(unexplored_actions)

    worker_id, move, build = action
    new_board = node.board.clone()

    new_worker = find_worker_by_id(new_board, worker_id)
    if new_worker is None:
        return None

    won = move_worker(new_board, new_worker, move)
    if won:
        num_players = len(game_config.player_ids)
        winning_vector = [-INF] * num_players
        winning_vector[node.player_index] = INF
        child = MCTSNode(new_board, node.player_index, parent=node, action=action)
        child.value = INF
        child.visits = 1
        node.children[action] = child
        return child

    build_block(new_board, new_worker, build)

    next_index = (node.player_index + 1) % len(game_config.player_ids)
    child = MCTSNode(new_board, next_index, parent=node, action=action)
    node.children[action] = child
    stats.bump()
    return child


def simulate(board, player_index, game_config, steps=4):
    """Fast heuristic-based simulation from the current position"""
    temp_board = board.clone()
    current_index = player_index

    for _ in range(steps):
        player_id = game_config.get_player_id(current_index)
        actions = generate_actions(temp_board, player_id)
        if not actions:
            break

        # Quick heuristic scoring WITHOUT expensive cloning
        best_action = None
        best_score = -INF

        for action in actions:
            worker_id, move, build = action
            w = find_worker_by_id(temp_board, worker_id)
            if w is None:
                continue

            # Quick win check without cloning
            move_height = temp_board.grid[move].height
            if move_height == 3:
                # This is a winning move for current player
                return INF if current_index == player_index else -INF

            # Fast heuristic: prioritize climbing + capping
            current_height = temp_board.grid[w.pos].height
            climb_bonus = (move_height - current_height) * 20
            cap_bonus = 100 if temp_board.grid[build].height == 3 else 0
            move_score = climb_bonus + cap_bonus + move_height * 5

            if move_score > best_score:
                best_score = move_score
                best_action = action

        if best_action is None:
            break

        worker_id, move, build = best_action
        temp_worker = find_worker_by_id(temp_board, worker_id)
        won = move_worker(temp_board, temp_worker, move)
        if won:
            return INF if current_index == player_index else -INF
        build_block(temp_board, temp_worker, build)
        current_index = (current_index + 1) % len(game_config.player_ids)

    player_id = game_config.get_player_id(player_index)
    return evaluate(temp_board, player_id)


def backpropagate(node, reward):
    """Backpropagate reward with player-relative perspective"""
    current_player = node.player_index
    while node:
        node.visits += 1
        # If node belongs to the current player add reward
        # If it's an opponent's node, subtract it (adversarial)
        if node.player_index == current_player:
            node.value += reward
        else:
            node.value -= reward
        node = node.parent


def mcts(board, depth, player_index, game_config, stats, iters=None, **kwargs):
    """
    MCTS wrapper that matches the maxn function signature for easy agent integration
    The depth parameter is converted to iterations
    """
    iterations = int(iters) if iters is not None else max(1000, depth * 400)
    return mcts_search(board, player_index, game_config, stats, iterations)


def generate_actions(board, player_id) -> List[Action]:
    actions = []
    workers = [w for w in board.workers if w.owner == player_id]
    for worker in workers:
        for move in legal_moves(board, worker.pos):
            for build in legal_builds(board, move):
                actions.append((worker.id, move, build))
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