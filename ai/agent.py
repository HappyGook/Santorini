import random

from game.rules import legal_moves, legal_builds


class Agent:
    def __init__(self, player_id: str):
        self.player_id = player_id
        self.phrases=[...]


    def make_move(self, board_state):
        move = self.decide_move(board_state)
        return move

    def build(self, board_state):
        build = self.decide_build(board_state)
        return build

    #Function that based on the agent's strategy (evaluation etc.) decides what move to make
    def decide_move(self, board_state):
        my_workers = [w for w in board_state.workers if w.owner == self.player_id]
        chosen_worker = random.choice(my_workers)
        moves = legal_moves(board_state, chosen_worker.pos)

        # Here some smart analysis will happen
        choice = random.choice(moves)
        return choice

    # Function that based on the agent's strategy (evaluation etc.) decides what build to make after the chosen move
    def decide_build(self, board_state):
        my_workers = [w for w in board_state.workers if w.owner == self.player_id]
        chosen_worker = random.choice(my_workers)
        builds = legal_builds(board_state, chosen_worker.pos)

        # Here some smart analysis will happen
        choice = random.choice(builds)
        return choice