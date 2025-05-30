import os
import sys
import yaml
import logging
import numpy as np
from agent.rl_agent import RLAgent
from env.btc_futures_env import BTCFuturesEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(config):
    """
    Load historical OHLCV data using the data_loader module.
    """
    from utils.data_loader import load_historical_data
    symbol = config['env']['symbol']
    timeframe = config['env']['timeframe']
    limit = config['env'].get('data_limit', 2000)
    df = load_historical_data(symbol, timeframe, limit)
    if df is None or df.empty:
        logger.error("No data loaded. Check symbol, timeframe, and API connectivity.")
        raise ValueError("No data loaded.")
    return df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/btc_futures.yaml')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        data = load_data(config)
        env = BTCFuturesEnv(config, data)
        agent = RLAgent(env, config)
        agent.load(args.model)
        obs = env.reset()
        done = False
        rewards = []
        equities = [env.equity]
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            equities.append(env.equity)
        logger.info(f"Backtest finished. Total reward: {np.sum(rewards):.4f}, Final equity: {equities[-1]:.4f}")
        # Optionally, save equity curve to logs
        np.save('logs/equity_curve.npy', np.array(equities))
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
