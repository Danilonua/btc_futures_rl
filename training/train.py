import os
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
    limit = config['env'].get('data_limit', 5000)
    df = load_historical_data(symbol, timeframe, limit)
    if df is None or df.empty:
        logger.error("No data loaded. Check symbol, timeframe, and API connectivity.")
        raise ValueError("No data loaded.")
    return df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/btc_futures.yaml')
    parser.add_argument('--save_path', type=str, default='models/best_model.zip')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        data = load_data(config)
        env = BTCFuturesEnv(config, data)
        agent = RLAgent(env, config)
        agent.train(config['agent']['total_timesteps'])
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        agent.save(args.save_path)
        logger.info('Training finished and model saved.')
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
