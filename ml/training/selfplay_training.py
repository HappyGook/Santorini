import random

from game.config import GameConfig
from gui.gameplay import GameController
from ml.selfplay import selfplay

controller = GameController
config = GameConfig(num_players=3)

print("Choose mode\n \t 1 for guided \n\t 2 for selfplay \n\t 3 for statistics\n\t 4 for manual setup")
mode = input()
if mode == "1":
    selfplay(controller, config,
             model_path="/Users/oleg77/PycharmProjects/PythonProject/Santorini/ml/learned_models/model2.pt",
             dataset_path="/Users/oleg77/PycharmProjects/PythonProject/Santorini/ml/datasets/guided_games.npz",
             num_games=1000, training_mode="guided")
elif mode == "2":
    selfplay(controller, config,
             model_path="/Users/oleg77/PycharmProjects/PythonProject/Santorini/ml/learned_models/model2.pt",
             dataset_path="/Users/oleg77/PycharmProjects/PythonProject/Santorini/ml/datasets/selfplay_tries1.npz",
             num_games=1000, training_mode="selfplay")
elif mode == "3":
    algos = ["mcts_NN", "ml", "maxn","mcts","minimax"]
    p1=random.choice(algos)
    algos.remove(p1)
    p2=random.choice(algos)
    algos.remove(p2)
    p3=random.choice(algos)
    print(f"Current algo settings are: P1:{p1}, P2:{p2}, P3:{p3}")
    selfplay(controller, config,
             model_path="/Users/oleg77/PycharmProjects/PythonProject/Santorini/ml/learned_models/model2.pt",
             dataset_path="/Users/oleg77/PycharmProjects/PythonProject/Santorini/ml/datasets/stats_games.npz",
             num_games=2, training_mode="statistics", p1_algo=p1, p2_algo=p2, p3_algo=p3)

elif mode == "4":
    # manual setup
    selfplay(controller, config,
             model_path="/Users/oleg77/PycharmProjects/PythonProject/Santorini/ml/learned_models/model2.pt",
             dataset_path="/Users/oleg77/PycharmProjects/PythonProject/Santorini/ml/datasets/stats_games.npz",
             num_games=2, training_mode="selfplay")

else:
    print("Invalid input >:(")