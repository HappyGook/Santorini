## Structure

# Santorini AI Project Structure

This document outlines the file structure for a Santorini AI game implementation with multiple AI approaches including classical algorithms, Monte Carlo Tree Search, and reinforcement learning.

## Project Directory Structure
Wished structure that we might want to have at some point
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
