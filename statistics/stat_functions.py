import json
import os


def normalize_score(score, min_val=-1000, max_val=1000):
    """Normalize score to [-1, 1] range."""
    return max(-1.0, min(1.0, (score - min_val) / (max_val - min_val) * 2 - 1))


def load_existing_statistics(stats_path):
    """Load existing statistics file or create new structure."""
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
    else:
        return {
            'total_games_played': 0,
            'algorithm_stats': {},
            'setup_positions': {},  # Setup position tracking
            'head_to_head': {},
            'algorithm_combinations': {}
        }


def update_setup_positions(statistics, setup_data):
    """Update setup position statistics."""
    if 'setup_positions' not in statistics:
        statistics['setup_positions'] = {}

    for player_id, algo, positions in setup_data:
        # Track setup positions for each algorithm
        if algo not in statistics['setup_positions']:
            statistics['setup_positions'][algo] = {}

        # Convert positions tuple to string for JSON serialization
        setup_key = f"{positions[0]}_{positions[1]}"

        # Initialize the key if it doesn't exist
        if setup_key not in statistics['setup_positions'][algo]:
            statistics['setup_positions'][algo][setup_key] = 0

        statistics['setup_positions'][algo][setup_key] += 1

def update_statistics(statistics, game_data):
    """Update statistics with new game data."""
    winner_algo = game_data['winner_algo']
    winner_player_id = game_data.get('winner_player_id')
    participants = game_data['participants']
    game_moves_data = game_data['moves_data']
    setup_data = game_data.get('setup_data', [])

    # Update total games played
    statistics['total_games_played'] += 1

    # Update setup positions
    if setup_data:
        update_setup_positions(statistics, setup_data)

    # Initialize algorithm stats if not present
    for player_id, algo in participants:
        if algo not in statistics['algorithm_stats']:
            statistics['algorithm_stats'][algo] = {
                'total_games': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_moves': 0,
                'up_moves': 0,
                'down_moves': 0,
                'same_level_moves': 0,
                'mean_score': 0.0,
                'score_sum': 0.0,
                'score_count': 0,
                'P1': {'games': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0},
                'P2': {'games': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0},
                'P3': {'games': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0}
            }

    # Update game counts and wins/losses for both overall and player-position stats
    for player_id, algo in participants:
        # Update overall algorithm stats
        stats = statistics['algorithm_stats'][algo]
        stats['total_games'] += 1

        # Update player position stats
        player_key = player_id  # Use 'P1', 'P2', 'P3' directly
        if player_key not in stats:
            stats[player_key] = {'games': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0}

        stats[player_key]['games'] += 1

        if algo == winner_algo and player_id == winner_player_id:
            stats['wins'] += 1
            stats[player_key]['wins'] += 1
        else:
            stats['losses'] += 1
            stats[player_key]['losses'] += 1

        # Update win rates
        stats['win_rate'] = stats['wins'] / stats['total_games']
        stats[player_key]['win_rate'] = stats[player_key]['wins'] / stats[player_key]['games']

    # Update move-specific data for both overall and player-position stats
    for algo_player, moves_info in game_moves_data.items():
        # Extract algorithm and player_id from the key
        if '_moves_' in algo_player:
            algo, player_id = algo_player.split('_moves_')

            # Update overall algorithm stats
            if algo in statistics['algorithm_stats']:
                stats = statistics['algorithm_stats'][algo]
                stats['total_moves'] += moves_info['total_moves']
                stats['up_moves'] += moves_info['up_moves']
                stats['down_moves'] += moves_info['down_moves']
                stats['same_level_moves'] += moves_info['same_level_moves']

                # Update mean score
                stats['score_sum'] += moves_info['score_sum']
                stats['score_count'] += moves_info['score_count']
                if stats['score_count'] > 0:
                    stats['mean_score'] = stats['score_sum'] / stats['score_count']

    # Update head-to-head records (keeping existing logic)
    if 'head_to_head' not in statistics:
        statistics['head_to_head'] = {}

    for player_id, algo in participants:
        if algo not in statistics['head_to_head']:
            statistics['head_to_head'][algo] = {}

        for other_player_id, other_algo in participants:
            if algo != other_algo:
                if other_algo not in statistics['head_to_head'][algo]:
                    statistics['head_to_head'][algo][other_algo] = {'wins': 0, 'losses': 0, 'games': 0}

                statistics['head_to_head'][algo][other_algo]['games'] += 1
                if algo == winner_algo:
                    statistics['head_to_head'][algo][other_algo]['wins'] += 1
                else:
                    statistics['head_to_head'][algo][other_algo]['losses'] += 1