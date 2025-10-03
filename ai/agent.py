import random

from game.rules import legal_moves, legal_builds
from game.board import BOARD_SIZE
from ai.heuristics import evaluate

class Agent:
    def __init__(self, player_id: str):
        self.player_id = player_id
        self.phrases=[...]


    def decide_action(self, board_state):
        my_workers = [w for w in board_state.workers if w.owner == self.player_id]
        # filter out workers that cannot move
        movable_workers = [w for w in my_workers if legal_moves(board_state, w.pos)]
        if not movable_workers:
            return None, None, None  # no legal moves at all

        chosen_worker = random.choice(movable_workers)
        moves = legal_moves(board_state, chosen_worker.pos)
        move_choice = random.choice(moves)
        if not moves:
            return None, None, None

        builds = legal_builds(board_state, move_choice)
        build_choice = random.choice(builds)
        if not builds:
            return None, None, None
        return chosen_worker, move_choice, build_choice

    def decide_setup(self, board_state):
        empty_cells = [
            (x, y)
            for x in range(BOARD_SIZE)
            for y in range(BOARD_SIZE)
            if board_state.get_cell((x, y)).worker_id is None
        ]
        return random.choice(empty_cells)

    def setup_workers(self, board_state):
        positions = []
        while len(positions) < 2:
            pos = self.decide_setup(board_state)
            if pos not in positions:
                positions.append(pos)
        return positions