from typing import List
from dataclasses import dataclass
from pathlib import Path

# === Layout ===
CELL = 80          # pixels per cell
MARGIN = 20        # outer padding

# === Colors ===
COLOR_MOVE = "lightblue"
COLOR_BUILD = "orange"
COLOR_SELECTED = "lightgreen"

PLAYER_COLORS = {
    "P1": "#FF6B6B",   # Red
    "P2": "#4ECDC4",   # Teal
    "P3": "#45B7D1",   # Blue
}

# === Asset Paths ===
ASSETS_DIR = Path(__file__).parent / "assets"
PLAYER_IMAGES = {
    "P1": ASSETS_DIR / "player_p1.png",
    "P2": ASSETS_DIR / "player_p2.png",
    "P3": ASSETS_DIR / "player_p3.png",
}
TILE_IMAGE = ASSETS_DIR / "tile.png"

MAX_WORKERS_PER_PLAYER = 2
@dataclass
class GameConfig:
    num_players: int
    workers_per_player: int = 2

    def __post_init__(self):
        self.player_ids = [f"P{i + 1}" for i in range(self.num_players)]
        self.total_workers = self.num_players * self.workers_per_player

    def get_worker_ids(self, player_index: int) -> List[str]:
        """Get worker IDs for a player (e.g., player 0 gets P1A, P1B)"""
        player_id = self.player_ids[player_index]
        return [f"{player_id}{chr(65 + i)}" for i in range(self.workers_per_player)]  # A, B, C...

    def next_player_index(self, current_index: int) -> int:
        """Get next player using modulo rotation"""
        return (current_index + 1) % self.num_players

    def get_player_id(self, index: int) -> str:
        """Convert index to player ID string"""
        if not self.player_ids:
            raise RuntimeError("GameConfig.player_ids is empty")

        idx = index % len(self.player_ids)
        return self.player_ids[idx]

    def get_player_index(self, player_id: str) -> int:
        """Convert player ID string to index"""
        return self.player_ids.index(player_id)

# Usage kinda like
# 2-player game: players [0,1] -> ["P1","P2"], workers ["P1A","P1B","P2A","P2B"]
# 3-player game: players [0,1,2] -> ["P1","P2","P3"], workers ["P1A","P1B","P2A","P2B","P3A","P3B"]