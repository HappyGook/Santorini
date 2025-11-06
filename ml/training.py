from game.config import GameConfig
from gui.gameplay import GameController
from ml.selfplay import selfplay


controller = GameController
config = GameConfig(num_players=3)
selfplay(controller, config, 1, "guided")