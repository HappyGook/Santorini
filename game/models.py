from dataclasses import dataclass
from typing import Optional, Tuple, List

from game.config import GameConfig

#constants

BOARD_SIZE = 5  #cell grid
MAX_LEVEL = 3   # tallest building level
DOME_LEVEL = 4  # BLOCKED cell level

Coord = Tuple[int, int]  # (row, col) coordinates 


@dataclass

class Cell:
    height: int = 0 #default
    worker_id: Optional[str] = None

@dataclass

class Worker:
    id: str # "P1A", "P1B", "P2A", "P2B"
    owner: str #player id P1 || P2
    pos: Optional[Coord] = None #row,col

def create_workers_for_game(game_config: GameConfig) -> List[Worker]:
    """Create all workers for a game based on configuration"""
    workers = []

    for player_index in range(game_config.num_players):
        player_id = game_config.get_player_id(player_index)
        worker_ids = game_config.get_worker_ids(player_index)

        for worker_id in worker_ids:
            workers.append(Worker(id=worker_id, owner=player_id, pos=None))

    return workers
