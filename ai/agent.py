import random

from game.rules import legal_moves, legal_builds
from game.board import BOARD_SIZE


class Agent:
    def __init__(self, player_id: str):
        self.player_id = player_id
        self.phrases=[...]


    def make_move(self, board_state):
        worker, move = self.decide_move(board_state)
        return worker, move

    def build(self, worker, board_state):
        build = self.decide_build(worker, board_state)
        return build

    #Function that based on the agent's strategy (evaluation etc.) decides what move to make
    def decide_move(self, board_state):
        my_workers = [w for w in board_state.workers if w.owner == self.player_id]
        # filter out workers that cannot move
        movable_workers = [w for w in my_workers if legal_moves(board_state, w.pos)]
        if not movable_workers:
            return None, None  # no legal moves at all

        chosen_worker = random.choice(movable_workers)
        moves = legal_moves(board_state, chosen_worker.pos)
        choice = random.choice(moves)
        return chosen_worker, choice

    # Function that based on the agent's strategy (evaluation etc.) decides what build to make after the chosen move
    def decide_build(self, worker, board_state):
        builds = legal_builds(board_state, worker.pos)

        # Here some smart analysis will happen
        choice = random.choice(builds)
        return choice

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