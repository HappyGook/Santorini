## Structure

# Santorini AI Project Structure
This document outlines the file structure for a Santorini AI game implementation with multiple AI approaches including Minimax, Max^n, Monte Carlo Tree Search

## Project Directory Structure
Wished structure that we might want to have at some point
```
santorini/
│
├── game/
│   ├── board.py        # Board representation & rules
│   ├── config.py       # Game configuration
│   ├── moves.py        # Move generation & validation
│   ├── models.py       # Data Classes 
│   └── models.py       # Game rules + Restrictions
│
├── ai/
│   ├── minimax.py      # Classical minimax with alpha-beta pruning
│   ├── maxn.py         # Max^n algorithm with deep pruning
│   ├── mcts.py         # Monte Carlo Tree Search
│   ├── agent.py        # Reflexive agent making moves based on an algorithm
│   └── heuristics.py   # Heuristic evaluation functions
│
├── gui/
│   ├── window.py       # GUI loop (tkinter)
│   ├── notation.py     # Chess-like game notation
│   ├── gameplay.py     # Game controller module
│   └── assets/         # Game assets
│
└── main.py             # Entry point to run the game
```
