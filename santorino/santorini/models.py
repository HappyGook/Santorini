from dataclasses import dataclass
from typing import Optional, Tuple

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