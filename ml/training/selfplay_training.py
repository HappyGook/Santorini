from game.config import GameConfig
from gui.gameplay import GameController
from ml.selfplay import selfplay


controller = GameController
config = GameConfig(num_players=3)
selfplay(controller, config,model_path="../learned_models/best.pt",
         dataset_path="../datasets/guided_games", num_games=300, training_mode="guided")