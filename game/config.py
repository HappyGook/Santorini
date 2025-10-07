from typing import List
from dataclasses import dataclass


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
        return self.player_ids[index]

    def get_player_index(self, player_id: str) -> int:
        """Convert player ID string to index"""
        return self.player_ids.index(player_id)

# Usage kinda like
# 2-player game: players [0,1] -> ["P1","P2"], workers ["P1A","P1B","P2A","P2B"]
# 3-player game: players [0,1,2] -> ["P1","P2","P3"], workers ["P1A","P1B","P2A","P2B","P3A","P3B"]