import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class StatisticsVisualizer:
    """Create graphs from json"""

    def __init__(self, json_path=None, output_dir=None):
        if json_path is None:
            json_path = Path(__file__).parent / "records" / "stats.json"
        if output_dir is None:
            output_dir = Path(__file__).parent / "records" / "graphs"

        self.json_path = Path(json_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load and parse the JSON data
        self.data = self._load_data()
        self.df_algorithms = self._create_dataframe()
        self.df_player_positions = self._create_player_position_dataframe()

        plt.style.use('default')
        # Set up color palette manually
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf']

    def _load_data(self):
        """Load JSON statistics data."""
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            print(f"Loaded statistics from {self.json_path}")
            return data
        except FileNotFoundError:
            print(f"Statistics file not found at {self.json_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return {}

    def _create_dataframe(self):
        """Create a dataframe from algorithm statistics."""
        if not self.data or 'algorithm_stats' not in self.data:
            return pd.DataFrame()

        rows = []
        for algo_name, stats in self.data['algorithm_stats'].items():
            # Main algorithm stats
            base_row = {
                'algorithm': algo_name,
                'player_position': 'Overall',
                'total_games': stats['total_games'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': stats['win_rate'],
                'total_moves': stats['total_moves'],
                'up_moves': stats['up_moves'],
                'down_moves': stats['down_moves'],
                'same_level_moves': stats['same_level_moves'],
                'mean_score': stats['mean_score'],
                'score_sum': stats['score_sum'],
                'score_count': stats['score_count']
            }
            rows.append(base_row)

            for pos in ['P1', 'P2', 'P3']:
                if pos in stats and stats[pos]['games'] > 0:
                    pos_row = {
                        'algorithm': algo_name,
                        'player_position': pos,
                        'total_games': stats[pos]['games'],
                        'wins': stats[pos]['wins'],
                        'losses': stats[pos]['losses'],
                        'win_rate': stats[pos]['win_rate'],
                        'total_moves': 0,
                        'up_moves': 0,
                        'down_moves': 0,
                        'same_level_moves': 0,
                        'mean_score': 0,
                        'score_sum': 0,
                        'score_count': 0
                    }
                    rows.append(pos_row)

        return pd.DataFrame(rows)

    def _create_player_position_dataframe(self):
        """dataframe specifically for player position analysis."""
        if not self.data or 'algorithm_stats' not in self.data:
            return pd.DataFrame()

        rows = []
        for algo_name, stats in self.data['algorithm_stats'].items():
            for pos in ['P1', 'P2', 'P3']:
                if pos in stats and stats[pos]['games'] > 0:
                    rows.append({
                        'algorithm': algo_name,
                        'player_position': pos,
                        'games': stats[pos]['games'],
                        'wins': stats[pos]['wins'],
                        'losses': stats[pos]['losses'],
                        'win_rate': stats[pos]['win_rate']
                    })

        return pd.DataFrame(rows)

    def _aggregate_setup_positions(self):
        """Aggregate setup positions across all algorithms since they use same heuristic."""
        if not self.data or 'setup_positions' not in self.data:
            return {}

        position_totals = defaultdict(int)
        for algo_name, positions in self.data['setup_positions'].items():
            for position_pair, count in positions.items():
                position_totals[position_pair] += count

        return dict(position_totals)

    def plot_algorithm_performance_overview(self):
        if self.df_algorithms.empty:
            return

        # Filter for overall stats only
        overall_stats = self.df_algorithms[self.df_algorithms['player_position'] == 'Overall'].copy()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Algorithm Performance Overview', fontsize=16, fontweight='bold')

        # Win Rate Comparison
        ax1 = axes[0, 0]
        bars = ax1.bar(overall_stats['algorithm'], overall_stats['win_rate'],
                       color=self.colors[:len(overall_stats)])
        ax1.set_title('Win Rate by Algorithm')
        ax1.set_ylabel('Win Rate')
        ax1.tick_params(axis='x', rotation=45)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom')

        # Total Games Played
        ax2 = axes[0, 1]
        ax2.bar(overall_stats['algorithm'], overall_stats['total_games'],
                color=self.colors[:len(overall_stats)])
        ax2.set_title('Total Games by Algorithm')
        ax2.set_ylabel('Total Games')
        ax2.tick_params(axis='x', rotation=45)

        # Move Distribution
        ax3 = axes[1, 0]
        move_types = ['up_moves', 'down_moves', 'same_level_moves']
        bottom = np.zeros(len(overall_stats))
        colors_moves = ['#2ca02c', '#d62728', '#1f77b4']
        for i, move_type in enumerate(move_types):
            ax3.bar(overall_stats['algorithm'], overall_stats[move_type],
                    bottom=bottom, label=move_type.replace('_', ' ').title(),
                    color=colors_moves[i])
            bottom += overall_stats[move_type]
        ax3.set_title('Move Type Distribution')
        ax3.set_ylabel('Number of Moves')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()

        # Mean Score
        ax4 = axes[1, 1]
        bars = ax4.bar(overall_stats['algorithm'], overall_stats['mean_score'],
                       color=self.colors[:len(overall_stats)])
        ax4.set_title('Mean Score by Algorithm')
        ax4.set_ylabel('Mean Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'algorithm_performance_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: algorithm_performance_overview.png")

    def plot_player_position_analysis(self):
        """Analyze performance by player position."""
        if self.df_player_positions.empty:
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Player Position Analysis', fontsize=16, fontweight='bold')

        pos_avg = self.df_player_positions.groupby('player_position')['win_rate'].mean()
        bars = ax.bar(pos_avg.index, pos_avg.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_title('Average Win Rate by Position (All Algorithms)')
        ax.set_ylabel('Average Win Rate')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'player_position_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: player_position_analysis.png")

    def plot_setup_position_analysis(self):
        """Analyze most popular setup positions overall (algorithm-agnostic)."""
        position_totals = self._aggregate_setup_positions()
        if not position_totals:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Setup Position Analysis (Overall)', fontsize=16, fontweight='bold')

        # Most popular setup positions overall
        ax1 = axes[0, 0]
        sorted_positions = sorted(position_totals.items(), key=lambda x: x[1], reverse=True)[:20]
        positions = [pos for pos, count in sorted_positions]
        counts = [count for pos, count in sorted_positions]

        y_pos = np.arange(len(positions))
        ax1.barh(y_pos, counts, color='#1f77b4')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([pos.replace('_', ' → ') for pos in positions], fontsize=8)
        ax1.set_title('Top 20 Most Popular Setup Positions')
        ax1.set_xlabel('Total Frequency')

        # Worker distance distribution
        ax2 = axes[0, 1]
        distances = []
        weights = []
        for position_pair, count in position_totals.items():
            try:
                pos1_str, pos2_str = position_pair.split('_')
                pos1 = eval(pos1_str)
                pos2 = eval(pos2_str)
                distance = np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
                distances.append(distance)
                weights.append(count)
            except (ValueError, SyntaxError):
                continue

        ax2.hist(distances, bins=15, weights=weights, alpha=0.7, color='#2ca02c')
        ax2.set_title('Worker Distance Distribution (Weighted)')
        ax2.set_xlabel('Distance Between Workers')
        ax2.set_ylabel('Frequency')

        # Heatmap of first worker positions
        ax3 = axes[1, 0]
        pos1_counts = defaultdict(int)
        for position_pair, count in position_totals.items():
            try:
                pos1_str, pos2_str = position_pair.split('_')
                pos1 = eval(pos1_str)
                pos1_counts[(pos1[0], pos1[1])] += count
            except (ValueError, SyntaxError):
                continue

        # Create 5x5 heatmap for first worker positions
        heatmap_data = np.zeros((5, 5))
        for (x, y), count in pos1_counts.items():
            if 0 <= x < 5 and 0 <= y < 5:
                heatmap_data[x, y] = count

        im = ax3.imshow(heatmap_data, cmap='Blues')
        ax3.set_title('First Worker Position Heatmap')
        ax3.set_xlabel('Column')
        ax3.set_ylabel('Row')

        # Add text annotations
        for i in range(5):
            for j in range(5):
                text = ax3.text(j, i, f'{int(heatmap_data[i, j])}',
                                ha="center", va="center",
                                color="white" if heatmap_data[i, j] > heatmap_data.max() / 2 else "black")
        plt.colorbar(im, ax=ax3)

        # Corner vs Center vs Edge analysis
        ax4 = axes[1, 1]
        corner_positions = [(0, 0), (0, 4), (4, 0), (4, 4)]
        edge_positions = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 4), (2, 0), (2, 4), (3, 0), (3, 4), (4, 1), (4, 2),
                          (4, 3)]
        center_positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]

        corner_count = 0
        edge_count = 0
        center_count = 0

        for position_pair, count in position_totals.items():
            try:
                pos1_str, pos2_str = position_pair.split('_')
                pos1 = eval(pos1_str)
                pos2 = eval(pos2_str)

                # Count both positions
                for pos in [pos1, pos2]:
                    if pos in corner_positions:
                        corner_count += count
                    elif pos in edge_positions:
                        edge_count += count
                    elif pos in center_positions:
                        center_count += count
            except (ValueError, SyntaxError):
                continue

        categories = ['Corner', 'Edge', 'Center']
        counts = [corner_count, edge_count, center_count]
        colors = ['#d62728', '#ff7f0e', '#2ca02c']

        bars = ax4.bar(categories, counts, color=colors)
        ax4.set_title('Position Type Preference')
        ax4.set_ylabel('Total Frequency')

        # Add percentage labels
        total = sum(counts)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = (height / total) * 100 if total > 0 else 0
            ax4.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                     f'{percentage:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'setup_position_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: setup_position_analysis.png")

    def plot_move_pattern_analysis(self):
        """Analyze movement patterns."""
        if self.df_algorithms.empty:
            return

        overall_stats = self.df_algorithms[self.df_algorithms['player_position'] == 'Overall'].copy()

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Movement Pattern Analysis', fontsize=16, fontweight='bold')

        # Climbing tendency (up moves ratio)
        ax1 = axes[0]
        up_ratios = overall_stats['up_moves'] / overall_stats['total_moves']
        bars = ax1.bar(overall_stats['algorithm'], up_ratios, color=self.colors[:len(overall_stats)])
        ax1.set_title('Climbing Tendency (Up Moves Ratio)')
        ax1.set_ylabel('Up Moves / Total Moves')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=up_ratios.mean(), color='red', linestyle='--', alpha=0.7, label='Average')
        ax1.legend()

        # Score vs Movement correlation
        ax2 = axes[1]
        ax2.scatter(up_ratios, overall_stats['mean_score'], s=100, alpha=0.7,
                    c=self.colors[:len(overall_stats)])
        for i, txt in enumerate(overall_stats['algorithm']):
            ax2.annotate(txt, (up_ratios.iloc[i], overall_stats['mean_score'].iloc[i]),
                         xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('Up Moves Ratio')
        ax2.set_ylabel('Mean Score')
        ax2.set_title('Climbing vs Performance')

        # Movement efficiency
        ax3 = axes[2]
        efficiency = overall_stats['mean_score'] / (overall_stats['total_moves'] / overall_stats['total_games'])
        bars = ax3.bar(overall_stats['algorithm'], efficiency, color=self.colors[:len(overall_stats)])
        ax3.set_title('Movement Efficiency\n(Score per Move per Game)')
        ax3.set_ylabel('Efficiency Score')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'move_pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: move_pattern_analysis.png")

    def generate_summary_report(self):
        """Generate a text summary of key findings."""
        if not self.data:
            return

        overall_stats = self.df_algorithms[self.df_algorithms['player_position'] == 'Overall'].copy()

        report_lines = [
            "SANTORINI GAME STATISTICS SUMMARY REPORT",
            "=" * 50,
            f"Total games analyzed: {self.data.get('total_games_played', 0)}",
            f"Algorithms analyzed: {len(overall_stats)}",
            "",
            "TOP PERFORMERS:",
        ]

        if not overall_stats.empty:
            # Best win rate
            best_winrate = overall_stats.loc[overall_stats['win_rate'].idxmax()]
            report_lines.extend([
                f"Highest win rate: {best_winrate['algorithm']} ({best_winrate['win_rate']:.3f})",
                f"Most games played: {overall_stats.loc[overall_stats['total_games'].idxmax()]['algorithm']} ({overall_stats['total_games'].max()} games)",
                f"Best mean score: {overall_stats.loc[overall_stats['mean_score'].idxmax()]['algorithm']} ({overall_stats['mean_score'].max():.4f})",
                "",
                "MOVEMENT PATTERNS:",
            ])

            # Movement analysis
            overall_stats_copy = overall_stats.copy()
            overall_stats_copy['up_ratio'] = overall_stats_copy['up_moves'] / overall_stats_copy['total_moves']
            most_aggressive = overall_stats_copy.loc[overall_stats_copy['up_ratio'].idxmax()]

            report_lines.extend([
                f"Most climbing-focused: {most_aggressive['algorithm']} ({most_aggressive['up_ratio']:.3f} up moves ratio)",
                f"Average up moves ratio: {overall_stats_copy['up_ratio'].mean():.3f}",
                "",
                "PLAYER POSITION INSIGHTS:",
            ])

            if not self.df_player_positions.empty:
                pos_performance = self.df_player_positions.groupby('player_position')['win_rate'].mean()
                best_pos = pos_performance.idxmax()
                worst_pos = pos_performance.idxmin()

                report_lines.extend([
                    f"Best performing position: {best_pos} (avg win rate: {pos_performance[best_pos]:.3f})",
                    f"Worst performing position: {worst_pos} (avg win rate: {pos_performance[worst_pos]:.3f})",
                ])

        # Setup position analysis
        position_totals = self._aggregate_setup_positions()
        if position_totals:
            most_popular = max(position_totals.items(), key=lambda x: x[1])
            report_lines.extend([
                "",
                "SETUP POSITION INSIGHTS:",
                f"Most popular setup: {most_popular[0]} ({most_popular[1]} times)",
                f"Total unique setups recorded: {len(position_totals)}"
            ])

        report_content = "\n".join(report_lines)

        # Save report
        report_path = self.output_dir / 'summary_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_content)

        print("SUMMARY REPORT:")
        print(report_content)
        print(f"\nReport saved to: {report_path}")

    def create_all_visualizations(self):
        """Generate all available visualizations."""
        print("Generating comprehensive statistics visualizations...")
        print(f"Data loaded: {len(self.data)} top-level keys")
        print(f"Output directory: {self.output_dir}")

        if not self.data:
            print("No data available for visualization!")
            return

        try:
            self.plot_algorithm_performance_overview()
            self.plot_player_position_analysis()
            self.plot_setup_position_analysis()
            self.plot_move_pattern_analysis()
            self.generate_summary_report()

            print(f"\nAll visualizations complete! Check {self.output_dir} for output files.")

        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()


def main():
    json_path = Path(__file__).parent / "records" / "stats.json"
    output_dir = Path(__file__).parent / "records" / "graphs"

    visualizer = StatisticsVisualizer(json_path, output_dir)
    visualizer.create_all_visualizations()


if __name__ == "__main__":
    main()