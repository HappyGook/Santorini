from game.config import GameConfig
from gui.gameplay import GameController
from ml.selfplay import selfplay


controller = GameController
config = GameConfig(num_players=3)
selfplay(controller, config,model_path="ml/learned_models/guided_model.pt",
         dataset_path="ml/datasets/selfplay_tries", num_games=10, training_mode="selfplay")