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
    parser.add_argument('--episodes', type=int, default=10)  # уменьшено для быстрого теста
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--test_mode', action='store_true')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        data = load_data(config)
        env = BTCFuturesEnv(config, data)
        agent = RLAgent(env, config)
        episode_rewards = []
        if args.test_mode:
            for episode in range(args.episodes):
                state, _ = env.reset()
                episode_reward = 0
                done = False
                steps = 0
                action_counts = {}
                while not done and steps < 50:  # ограничение длины эпизода
                    action = agent.model.action_space.sample()
                    action = int(action)
                    action_counts[action] = action_counts.get(action, 0) + 1
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    if np.isnan(reward):
                        print(f"NaN reward detected at episode {episode+1}, step {steps+1}")
                        break
                    episode_reward += reward
                    state = next_state
                    done = terminated or truncated
                    steps += 1
                episode_rewards.append(episode_reward)
                epsilon = getattr(agent, 'epsilon', None)
                epsilon_str = f", Epsilon={epsilon:.3f}" if epsilon is not None else ""
                print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Portfolio={env.equity:.2f}{epsilon_str}, Steps={steps}, Actions={action_counts}")
        else:
            total_timesteps = args.episodes * 50
            agent.train(total_timesteps=total_timesteps)
            agent.save(args.save_path)
        logger.info('Controlled training finished.')
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
