from ai.agent import Agent
from game.board import Board
from game.config import GameConfig
from game.models import create_workers_for_game
from gui.gameplay import GameController
from gui.window import SantoriniTk, choose_mode_ui, build_players, place_workers_for_setup


def main():
    # Get game mode and number of players
    mode_selection = choose_mode_ui()  # Returns dict with mode and num_players

    # Create game configuration
    game_config = GameConfig(num_players=mode_selection["num_players"])

    # Create workers based on game configuration
    workers = create_workers_for_game(game_config)

    # Create board with configuration
    board = Board(game_config, workers)

    # Place workers in starting positions
    place_workers_for_setup(board, game_config)

    # Build player configuration
    players = build_players(mode_selection, game_config)

    # Create game controller
    controller = GameController(board, players, game_config)

    # Start GUI
    app = SantoriniTk(board, controller, game_config)
    app.mainloop()


if __name__ == "__main__":
    main()