from ai.agent import Agent
from game.board import Board
from gui.notation import GameNotation
from gui.gameplay import setup_workers, play_turn, render_board


def main():
    print("Welcome to Santorini CLI!")
    board = Board()
    notation = GameNotation()

    mode = input("Choose mode: (1) Human vs Human, (2) Human vs AI, (3) AI vs AI: ").strip()

    agents = {}
    if mode == "2":
        agents["P2"] = Agent("P2")
    elif mode == "3":
        agents["P1"] = Agent("P1")
        agents["P2"] = Agent("P2")


    # Setup phase
    setup_workers(board, notation, agents)
    render_board(board)

    # Default starting player if not set
    if not getattr(board, 'current_player', None):
        board.current_player = "P1"

    # Gameplay loop
    while True:
        game_over, worker, dst = play_turn(board, notation, ai_agents=agents)
        render_board(board)

        if game_over:
            notation.save()
            print("Game over!")
            print("Final notation:")
            print(notation)
            break


if __name__ == "__main__":
    main()