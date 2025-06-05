import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from agent.rl_agent import RLAgent
from env.btc_futures_env import BTCFuturesEnv
from stable_baselines3.common.callbacks import EvalCallback
from utils.data_loader import load_historical_data

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def walk_forward_validation(config, save_path, n_chunks=5, eval_episodes=1):
    data = load_historical_data(config['env']['symbol'], config['env']['timeframe'], config['env'].get('data_limit', 5000))
    chunk_size = len(data) // n_chunks
    all_results = []
    for i in range(n_chunks-1):
        train = data.iloc[i*chunk_size:(i+3)*chunk_size]
        val = data.iloc[(i+3)*chunk_size:(i+4)*chunk_size]
        print(f"Walk-forward chunk {i+1}: Train {train.shape}, Val {val.shape}")
        env = BTCFuturesEnv(config, train)
        val_env = BTCFuturesEnv(config, val)
        agent = RLAgent(env, config)
        eval_callback = EvalCallback(val_env, best_model_save_path="./logs/best_model_{}/".format(i),
                                     log_path="./logs/eval_{}/".format(i), eval_freq=1000,
                                     n_eval_episodes=eval_episodes, deterministic=True, render=False)
        total_timesteps = config['agent'].get('total_timesteps', 100000)
        agent.model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        agent.save(os.path.join(save_path, f"best_model_chunk_{i}.zip"))
        all_results.append({
            'chunk': i,
            'train_start': train.index[0],
            'train_end': train.index[-1],
            'val_start': val.index[0],
            'val_end': val.index[-1],
        })
    return all_results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/btc_futures.yaml')
    parser.add_argument('--save_path', type=str, default='models/')
    parser.add_argument('--n_chunks', type=int, default=5)
    parser.add_argument('--eval_episodes', type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    config = load_config(args.config)
    walk_forward_validation(config, args.save_path, n_chunks=args.n_chunks, eval_episodes=args.eval_episodes)

if __name__ == "__main__":
    main()
