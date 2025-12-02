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
            'head_to_head': {},
            'algorithm_combinations': {}
        }


def update_statistics(statistics, game_data):
    """Update statistics with new game data."""
    winner_algo = game_data['winner_algo']
    participants = game_data['participants']
    game_moves_data = game_data['moves_data']

    # Update total games played
    statistics['total_games_played'] += 1

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
                'score_count': 0
            }

    # Update game counts and wins/losses
    for player_id, algo in participants:
        stats = statistics['algorithm_stats'][algo]
        stats['total_games'] += 1

        if algo == winner_algo:
            stats['wins'] += 1
        else:
            stats['losses'] += 1

        # Update win rate
        stats['win_rate'] = stats['wins'] / stats['total_games']

    # Update move-specific data
    for algo, moves_info in game_moves_data.items():
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

    # Update head-to-head records
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

    # Update algorithm combinations
    if 'algorithm_combinations' not in statistics:
        statistics['algorithm_combinations'] = {}

    combo_key = '_vs_'.join(sorted([algo for _, algo in participants]))
    if combo_key not in statistics['algorithm_combinations']:
        statistics['algorithm_combinations'][combo_key] = {'games': 0, 'winners': {}}

    statistics['algorithm_combinations'][combo_key]['games'] += 1
    if winner_algo not in statistics['algorithm_combinations'][combo_key]['winners']:
        statistics['algorithm_combinations'][combo_key]['winners'][winner_algo] = 0
    statistics['algorithm_combinations'][combo_key]['winners'][winner_algo] += 1