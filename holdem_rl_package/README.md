# Gymnasium No-Limit Hold'em RL Engine & Interactive Arena

## Contents
1. `holdem_rl_arena.html` — Self-contained interactive 4-8 player No-Limit Hold'em web interface with embedded Dueling Double DQN RL policy, live 24-dim Gymnasium state telemetry, in-browser self-play trainer, and 1-click artifact exporters.
2. `holdem_gym_rl.py` — Standalone Python OpenAI Gymnasium-compatible environment (`HoldemGymEnv`) and pure-NumPy Dueling Double DQN + Counterfactual Advantage trainer (`DuelingDQNAgent`).
3. `rl_weights.json` — Pre-trained 24x64x48x7 Dueling Double DQN weights and 6-max benchmark metrics across TAG, LAG, SOLID, CALL, and NIT opponent personas.

## Running Locally
- Open `holdem_rl_arena.html` in any modern browser to play against the RL agent or run live self-play episodes.
- Run `python3 holdem_gym_rl.py` to train a new agent in Python and export updated weights to `rl_weights.json`.
