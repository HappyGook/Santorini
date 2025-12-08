from game.board import Board
from game.config import GameConfig
from gui.gameplay import GameController
from gui.window import SantoriniTk, choose_mode_ui, build_players

def main():
    while True:
        mode_selection = choose_mode_ui()
        game_config = GameConfig(num_players=mode_selection["num_players"])

        # Empty board — GUI setup will place all workers
        board = Board(game_config, workers=[])

        players = build_players(mode_selection, game_config)
        controller = GameController(board, players, game_config)

        app = SantoriniTk(board, controller, game_config)
        app.mainloop()

if __name__ == "__main__":
    main()


    