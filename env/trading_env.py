import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Tuple
import logging
from utils.risk_management import calculate_var, apply_stop_loss
from utils.pattern_logger import log_pattern

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """
    A reinforcement learning environment for BTC futures trading.
    
    This environment simulates trading on BTC/USD futures with realistic
    transaction costs, risk management, and position constraints.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df: pd.DataFrame, config: Dict):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary with environment parameters
        """
        super(TradingEnv, self).__init__()
        
        # Store configuration
        self.config = config
        self.env_config = config['env']
        
        # Data setup
        self.df = df
        self.add_features()
        self.current_step = 0
        self.total_steps = len(self.df) - 1
        
        # Trading parameters
        self.initial_balance = self.env_config.get('initial_balance', 10000)
        self.balance = self.initial_balance
        self.max_position = self.env_config.get('max_position', 1.0)
        self.max_leverage = self.env_config.get('max_leverage', 5)
        self.transaction_fee = self.env_config.get('transaction_fee', 0.0004)  # 0.04% fee
        self.slippage = self.env_config.get('slippage', 0.0001)  # 0.01% slippage
        
        # Risk management
        self.stop_loss = self.env_config.get('stop_loss', 0.02)  # 2% stop loss
        self.take_profit = self.env_config.get('take_profit', 0.05)  # 5% take profit
        
        # Position tracking
        self.position = 0  # -1 (short) to 1 (long), 0 is neutral
        self.position_history = []
        self.entry_price = 0
        
        # Performance tracking
        self.trades = []
        self.returns = []
        self.cumulative_returns = []
        self.portfolio_values = [self.initial_balance]
        
        # Define action and observation spaces
        # Actions: -1 (short), 0 (hold/close), 1 (long)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price data + technical indicators + position info
        num_features = self.df.shape[1] + 2  # +2 for position and entry price
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )
        
        logger.info(f"TradingEnv initialized with {len(df)} data points")
    
    def add_features(self):
        """Add technical indicators and features to the dataframe."""
        df = self.df.copy()
        
        # Add basic technical indicators
        # SMA
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_25'] = df['close'].rolling(window=25).mean()
        
        # EMA
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * std
        df['bb_lower'] = df['bb_middle'] - 2 * std
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        # Normalize features
        for col in df.columns:
            if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        # Fill NaN values
        df.fillna(0, inplace=True)
        
        self.df = df
        logger.debug(f"Added features to dataframe. Shape: {df.shape}")
    
    def _get_observation(self):
        """
        Get the current observation (state) from the environment.
        
        Returns:
            numpy array with the current observation
        """
        # Get current OHLCV and indicators
        obs = self.df.iloc[self.current_step].values
        
        # Add position and entry price information
        position_info = np.array([self.position, self.entry_price])
        
        return np.concatenate([obs, position_info]).astype(np.float32)
    
    def _take_action(self, action):
        """
        Execute the trading action and calculate reward.
        
        Args:
            action: 0 (short), 1 (hold/close), 2 (long)
        
        Returns:
            float: reward from the action
        """
        # Map action from [0, 1, 2] to [-1, 0, 1]
        action_map = {0: -1, 1: 0, 2: 1}
        mapped_action = action_map[action]
        
        # Get current price data
        current_price = self.df.iloc[self.current_step]['close']
        
        # Calculate reward (profit/loss)
        reward = 0
        
        # Check if we need to apply stop loss or take profit
        if self.position != 0:
            self.position = apply_stop_loss(
                self.position, 
                current_price / self.entry_price, 
                self.stop_loss, 
                self.take_profit
            )
            
            # If position was closed by stop loss or take profit
            if self.position == 0:
                pnl = self._calculate_pnl(current_price)
                reward = pnl
                self.balance += pnl
                self.trades.append({
                    'entry_step': self.position_history[-1][0],
                    'exit_step': self.current_step,
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'position': self.position_history[-1][1],
                    'pnl': pnl,
                    'reason': 'stop_loss_or_take_profit'
                })
                self.entry_price = 0
        
        # Process the new action
        if mapped_action != 0:  # If not hold
            # Close existing position if any
            if self.position != 0:
                pnl = self._calculate_pnl(current_price)
                reward = pnl
                self.balance += pnl
                self.trades.append({
                    'entry_step': self.position_history[-1][0],
                    'exit_step': self.current_step,
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'position': self.position_history[-1][1],
                    'pnl': pnl,
                    'reason': 'position_change'
                })
            
            # Open new position
            self.position = mapped_action * self.max_position
            self.entry_price = current_price
            self.position_history.append((self.current_step, self.position))
            
            # Apply transaction costs
            transaction_cost = abs(self.position) * current_price * (self.transaction_fee + self.slippage)
            self.balance -= transaction_cost
            reward -= transaction_cost
        
        elif mapped_action == 0 and self.position != 0:  # Close position
            pnl = self._calculate_pnl(current_price)
            reward = pnl
            self.balance += pnl
            self.trades.append({
                'entry_step': self.position_history[-1][0],
                'exit_step': self.current_step,
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'position': self.position_history[-1][1],
                'pnl': pnl,
                'reason': 'close_position'
            })
            self.position = 0
            self.entry_price = 0
        
        # Calculate portfolio value
        portfolio_value = self.balance
        if self.position != 0:
            portfolio_value += self._calculate_pnl(current_price)
        
        self.portfolio_values.append(portfolio_value)
        
        # Calculate returns
        if len(self.portfolio_values) > 1:
            current_return = (self.portfolio_values[-1] / self.portfolio_values[-2]) - 1
            self.returns.append(current_return)
            
            if len(self.cumulative_returns) > 0:
                self.cumulative_returns.append((1 + self.cumulative_returns[-1]) * (1 + current_return) - 1)
            else:
                self.cumulative_returns.append(current_return)
        
        # Log pattern if configured
        if self.config.get('pattern_logging', {}).get('enabled', False):
            window_size = self.config['pattern_logging'].get('window_size', 20)
            start_idx = max(0, self.current_step - window_size)
            window_data = self.df.iloc[start_idx:self.current_step + 1]
            log_pattern(window_data, self.config)
        
        return reward
    
    def _calculate_pnl(self, current_price):
        """
        Calculate profit/loss for the current position.
        
        Args:
            current_price: Current price to calculate PnL against entry price
            
        Returns:
            float: Profit/loss amount
        """
        if self.position == 0 or self.entry_price == 0:
            return 0
        
        price_diff = current_price - self.entry_price
        pnl = self.position * price_diff
        
        # Apply transaction fee for closing
        transaction_cost = abs(self.position) * current_price * (self.transaction_fee + self.slippage)
        
        return pnl - transaction_cost
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Execute action and get reward
        reward = self._take_action(action)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.total_steps
        
        # Get new observation
        obs = self._get_observation() if not done else None
        
        # Prepare info dict
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self.portfolio_values[-1],
            'step': self.current_step,
        }
        
        if done:
            # Calculate final metrics
            info['total_trades'] = len(self.trades)
            info['profitable_trades'] = sum(1 for trade in self.trades if trade['pnl'] > 0)
            info['total_profit'] = sum(trade['pnl'] for trade in self.trades)
            info['final_balance'] = self.balance
            info['max_drawdown'] = self._calculate_max_drawdown()
            info['sharpe_ratio'] = self._calculate_sharpe_ratio()
            
            logger.info(f"Episode finished. Final balance: {self.balance:.2f}, "
                       f"Total profit: {info['total_profit']:.2f}, "
                       f"Trades: {info['total_trades']}, "
                       f"Win rate: {info['profitable_trades'] / max(1, info['total_trades']):.2%}")
        
        return obs, reward, done, info
    
    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            numpy array: Initial observation
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.position_history = []
        self.trades = []
        self.returns = []
        self.cumulative_returns = []
        self.portfolio_values = [self.initial_balance]
        
        return self._get_observation()
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode != 'human':
            raise NotImplementedError(f"Render mode {mode} not implemented")
        
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position: {self.position}")
        print(f"Entry Price: ${self.entry_price:.2f}")
        print(f"Current Price: ${self.df.iloc[self.current_step]['close']:.2f}")
        print(f"Portfolio Value: ${self.portfolio_values[-1]:.2f}")
        if self.trades:
            print(f"Last Trade PnL: ${self.trades[-1]['pnl']:.2f}")
    
    def _calculate_max_drawdown(self):
        """
        Calculate maximum drawdown from portfolio values.
        
        Returns:
            float: Maximum drawdown as a percentage
        """
        portfolio_values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return abs(min(drawdown))
    
    def _calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """
        Calculate Sharpe ratio from returns.
        
        Args:
            risk_free_rate: Risk-free rate (default: 0)
            
        Returns:
            float: Sharpe ratio
        """
        if not self.returns:
            return 0
        
        returns = np.array(self.returns)
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-9) * np.sqrt(252)  # Annualized