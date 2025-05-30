# BTC Futures RL Agent

A modular reinforcement learning agent for algorithmic trading on BTC/USD futures contracts.

## Features
- RL agent (PPO/DQN/SAC via stable-baselines3 or RLlib)
- Custom trading environment with realistic reward (profit minus fees/slippage)
- Risk management (max position, leverage, stop-loss, take-profit, VaR)
- Pattern knowledge base logging
- Modular codebase: `data/`, `env/`, `agent/`, `training/`, `utils/`, `logs/`
- Unit tests and CI (GitHub Actions)

## Installation
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## API Key Setup
- Place your exchange API keys in `configs/btc_futures.yaml` (see example).

## Training
```bash
python training/train.py --config configs/btc_futures.yaml
```

## Backtesting
```bash
python backtest.py --model models/best_model.zip --config configs/btc_futures.yaml
```

## Testing
```bash
pytest
```

## Style Checks
```bash
flake8 .
black --check .
```

---

## Directory Structure
```
agent/           # RL agent wrappers
backtest.py      # Backtesting script
configs/         # Config files (YAML)
data/            # Data loading & streaming
env/             # Trading environment
logs/            # Logs & pattern knowledge base
requirements.txt # Dependencies
training/        # Training script
utils/           # Utilities
.github/         # CI workflows
tests/           # Unit tests
```
