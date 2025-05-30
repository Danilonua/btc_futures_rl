import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from stable_baselines3 import PPO
from env.trading_env import TradingEnv
from utils.data_loader import load_historical_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def backtest(model_path, config_path, plot=True):
    """
    Backtest a trained model on historical data.
    
    Args:
        model_path: Path to the saved model
        config_path: Path to the configuration file
        plot: Whether to plot the results
    
    Returns:
        dict: Backtest results
    """
    # Load configuration
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return
    
    # Load test data
    try:
        logger.info(f"Loading test data for {config['env']['symbol']} {config['env']['timeframe']}")
        df = load_historical_data(
            symbol=config['env']['symbol'],
            timeframe=config['env']['timeframe'],
            limit=config['backtest'].get('data_limit', 1000)
        )
        logger.info(f"Successfully loaded {len(df)} records")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return
    
    # Create environment
    try:
        env = TradingEnv(df, config)
        logger.info("Trading environment created successfully")
    except Exception as e:
        logger.error(f"Environment creation failed: {e}")
        return
    
    # Load model
    try:
        model = PPO.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return
    
    # Run backtest
    try:
        obs = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        # Get results
        results = {
            'final_balance': info['final_balance'],
            'total_profit': info['total_profit'],
            'total_trades': info['total_trades'],
            'profitable_trades': info['profitable_trades'],
            'win_rate': info['profitable_trades'] / max(1, info['total_trades']),
            'max_drawdown': info['max_drawdown'],
            'sharpe_ratio': info['sharpe_ratio'],
            'portfolio_values': env.portfolio_values,
            'returns': env.returns,
            'cumulative_returns': env.cumulative_returns,
            'trades': env.trades
        }
        
        # Print results
        logger.info("Backtest Results:")
        logger.info(f"Final Balance: ${results['final_balance']:.2f}")
        logger.info(f"Total Profit: ${results['total_profit']:.2f}")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.2%}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
        # Plot results
        if plot:
            plot_backtest_results(df, results, config['env']['symbol'])
        
        return results
    
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return None


def plot_backtest_results(df, results, symbol):
    """
    Plot backtest results.
    
    Args:
        df: DataFrame with price data
        results: Backtest results dictionary
        symbol: Trading symbol
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Portfolio Value
    plt.subplot(3, 1, 1)
    plt.plot(results['portfolio_values'])
    plt.title(f'Portfolio Value - {symbol}')
    plt.grid(True)
    
    # Plot 2: Cumulative Returns
    plt.subplot(3, 1, 2)
    plt.plot(results['cumulative_returns'])
    plt.title('Cumulative Returns')
    plt.grid(True)
    
    # Plot 3: Price with Buy/Sell Signals
    plt.subplot(3, 1, 3)
    plt.plot(df['close'])
    
    # Add buy/sell markers
    for trade in results['trades']:
        entry_idx = trade['entry_step']
        exit_idx = trade['exit_step']
        
        if trade['position'] > 0:  # Long position
            plt.scatter(entry_idx, df['close'].iloc[entry_idx], color='green', marker='^', s=100)
            plt.scatter(exit_idx, df['close'].iloc[exit_idx], color='red', marker='v', s=100)
        else:  # Short position
            plt.scatter(entry_idx, df['close'].iloc[entry_idx], color='red', marker='v', s=100)
            plt.scatter(exit_idx, df['close'].iloc[exit_idx], color='green', marker='^', s=100)
    
    plt.title(f'Price Chart with Trades - {symbol}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'logs/backtest_{symbol.replace("/", "_")}.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtest RL trading agent')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')
    
    args = parser.parse_args()
    
    backtest(args.model, args.config, not args.no_plot)