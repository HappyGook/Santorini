## Structure

# Santorini AI Project Structure

This document outlines the file structure for a Santorini AI game implementation with multiple AI approaches including classical algorithms, Monte Carlo Tree Search, and reinforcement learning.

## Project Directory Structure

```
santorini-ai/
│
├── game/
│   ├── board.py        # Board representation & rules
│   ├── moves.py        # Move generation & validation
│   ├── state.py        # Game state class
│   └── utils.py
│
├── ai/
│   ├── minimax.py      # Classical minimax with alpha-beta pruning
│   ├── mcts.py         # Monte Carlo Tree Search
│   ├── rl_agent.py     # Reinforcement learning (PyTorch model)
│   └── evaluation.py   # Heuristic evaluation functions
│
├── gui/
│   ├── gui.py          # GUI loop (pygame/tkinter)
│   └── assets/         # (if you use icons, images)
│
├── training/
│   ├── self_play.py    # Generate training data
│   ├── train.py        # Training loop for RL
│   └── replay_buffer.py
│
├── tests/
│   ├── test_board.py
│   ├── test_ai.py
│   └── ...
│
├── main.py             # Entry point to run the game
└── requirements.txt
```

## Module Descriptions

### Game Module (`game/`)
Contains the core game logic and mechanics for Santorini.

- **`board.py`** - Handles the 5x5 game board representation, building placement, and game rules enforcement
- **`moves.py`** - Generates valid moves for players and validates move legality
- **`state.py`** - Manages the complete game state including player positions, building heights, and turn tracking
- **`utils.py`** - Utility functions and helper methods used across the game module

### AI Module (`ai/`)
Implements various artificial intelligence approaches for playing Santorini.

- **`minimax.py`** - Classical minimax algorithm with alpha-beta pruning for optimal play
- **`mcts.py`** - Monte Carlo Tree Search implementation for strategic decision making
- **`rl_agent.py`** - Reinforcement learning agent using PyTorch neural networks
- **`evaluation.py`** - Heuristic evaluation functions for position assessment

### GUI Module (`gui/`)
Handles the graphical user interface and visual presentation.

- **`gui.py`** - Main GUI application loop using pygame or tkinter
- **`assets/`** - Directory for game assets like icons, images, and visual resources

### Training Module (`training/`)
Supports machine learning model training and data generation.

- **`self_play.py`** - Generates training data through AI vs AI self-play games
- **`train.py`** - Main training loop for reinforcement learning models
- **`replay_buffer.py`** - Experience replay buffer for storing and sampling training data

### Tests Module (`tests/`)
Unit tests and integration tests for code quality assurance.

- **`test_board.py`** - Tests for board logic and game rules
- **`test_ai.py`** - Tests for AI algorithm implementations
- Additional test files as needed

### Root Files
- **`main.py`** - Main entry point to launch the Santorini game
- **`requirements.txt`** - Python package dependencies for the project

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the game: `python main.py`
3. Run tests: `python -m pytest tests/`

## AI Approaches

This implementation supports multiple AI strategies:

1. **Minimax with Alpha-Beta Pruning** - Optimal play through exhaustive search with pruning
2. **Monte Carlo Tree Search (MCTS)** - Probabilistic search using random simulations
3. **Reinforcement Learning** - Neural network trained through self-play experience

Each approach offers different strengths in terms of playing style, computational requirements, and learning capabilities.