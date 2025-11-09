from game.config import GameConfig
from gui.gameplay import GameController
from ml.selfplay import selfplay


controller = GameController
config = GameConfig(num_players=3)
selfplay(controller, config,model_path="../learned_models/guided_model.pt",
         dataset_path="../datasets/selfplayed_games", num_games=100, training_mode="selfplay")