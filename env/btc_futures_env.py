import gymnasium as gym
import numpy as np
from gymnasium import spaces
import logging
import pandas as pd
from utils.pattern_logger import PatternLogger

class BTCFuturesEnv(gym.Env):
    """
    Custom Environment for BTC/USDT Futures trading compatible with OpenAI Gym.
    Reward = realized profit - fees - slippage.
    Implements risk management (max position, leverage, stop-loss, take-profit, VaR).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config, data):
        super(BTCFuturesEnv, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.data = data
        self.fee = config.get('fee', 0.0004)  # 0.04% default
        self.slippage = config.get('slippage', 0.0002)  # 0.02% default
        self.max_position = config.get('max_position', 1.0)  # 1 BTC
        self.max_leverage = config.get('max_leverage', 5)
        self.stop_loss = config.get('stop_loss', 0.02)  # 2%
        self.take_profit = config.get('take_profit', 0.04)  # 4%
        self.var_window = config.get('var_window', 30)
        self.var_quantile = config.get('var_quantile', 0.01)
        self.initial_balance = config.get('initial_balance', 10000)
        self.symbol = config.get('symbol', 'BTC/USDT')
        self.timeframe = config.get('timeframe', '1m')
        self.pattern_logger = PatternLogger(backend='json', path='logs/patterns.json')
        self._setup_spaces()
        self.reset()

    def _setup_spaces(self):
        # Observation: OHLCV + position + balance
        obs_len = 6  # [open, high, low, close, volume, position]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32
        )
        # Actions: 0 = hold, 1 = long, 2 = short, 3 = close
        self.action_space = spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # BTC
        self.entry_price = 0.0
        self.equity = self.initial_balance
        self.done = False
        self.trades = []
        self.portfolio_values = [self.initial_balance]
        self.returns = []
        self.cumulative_returns = []
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        obs = np.array([
            row['open'], row['high'], row['low'], row['close'], row['volume'], self.position
        ], dtype=np.float32)
        return obs

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, False, {}
        row = self.data.iloc[self.current_step]
        price = row['close']
        reward = 0.0
        info = {}
        # Action logic
        if action == 1:  # Long
            if self.position == 0:
                self.position = min(self.max_position, self.balance * self.max_leverage / price)
                self.entry_price = price
                self._log_trade('long', price)
        elif action == 2:  # Short
            if self.position == 0:
                self.position = -min(self.max_position, self.balance * self.max_leverage / price)
                self.entry_price = price
                self._log_trade('short', price)
        elif action == 3:  # Close
            if self.position != 0:
                pnl = (price - self.entry_price) * self.position
                fee = abs(self.position) * price * self.fee
                slip = abs(self.position) * price * self.slippage
                reward = pnl - fee - slip
                self.balance += reward
                self.position = 0
                self.entry_price = 0
                self._log_trade('close', price, reward)
        # Risk management: stop-loss/take-profit
        if self.position != 0:
            pnl_pct = (price - self.entry_price) / self.entry_price * np.sign(self.position)
            if pnl_pct <= -self.stop_loss or pnl_pct >= self.take_profit:
                pnl = (price - self.entry_price) * self.position
                fee = abs(self.position) * price * self.fee
                slip = abs(self.position) * price * self.slippage
                reward = pnl - fee - slip
                self.balance += reward
                self.position = 0
                self.entry_price = 0
                self._log_trade('auto_close', price, reward)
        # VaR risk control
        if self.current_step >= self.var_window:
            returns = self.data['close'].pct_change().iloc[self.current_step-self.var_window:self.current_step]
            var = np.quantile(returns, self.var_quantile)
            info['VaR'] = var
            if var < -self.stop_loss:
                self.done = True
                info['risk_stop'] = True
        self.equity = self.balance + (price - self.entry_price) * self.position if self.position != 0 else self.balance
        self.portfolio_values.append(self.equity)
        self.returns.append(self.equity / self.portfolio_values[-2] - 1 if len(self.portfolio_values) > 1 else 0)
        self.cumulative_returns.append(np.prod([1 + r for r in self.returns]) - 1)
        # Pattern detection and logging
        self._detect_and_log_patterns()
        self.current_step += 1
        if self.current_step >= len(self.data) - 1 or self.equity <= 0:
            self.done = True
        terminated = self.done
        truncated = False
        return self._get_observation(), reward, terminated, truncated, info

    def _log_trade(self, side, price, reward=0.0):
        self.trades.append({
            'step': self.current_step,
            'side': side,
            'price': price,
            'position': self.position,
            'reward': reward
        })

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}, Equity: {self.equity:.2f}")

    def _detect_and_log_patterns(self):
        # Example: Double bottom detection (very simple, for illustration)
        window = 20
        if self.current_step > window:
            closes = self.data['close'].iloc[self.current_step-window:self.current_step].values
            min_idx = np.argmin(closes)
            if 2 < min_idx < window-3:
                # Check for double bottom: two local minima within window
                left = closes[:min_idx]
                right = closes[min_idx+1:]
                if len(left) > 0 and len(right) > 0:
                    if np.min(left) > closes[min_idx] and np.min(right) > closes[min_idx]:
                        confidence = 0.8
                        example_candles = self.data.iloc[self.current_step-window:self.current_step].reset_index().to_dict('records')
                        self.pattern_logger.log_pattern(
                            name='double_bottom',
                            time_window=f"{self.data.index[self.current_step-window]} - {self.data.index[self.current_step-1]}",
                            confidence=confidence,
                            example_candles=example_candles
                        )
        # Example: RSI overbought/oversold
        rsi_period = 14
        if self.current_step > rsi_period:
            closes = self.data['close'].iloc[self.current_step-rsi_period:self.current_step]
            delta = closes.diff().dropna()
            up = delta.clip(lower=0).mean()
            down = -delta.clip(upper=0).mean()
            rs = up / down if down != 0 else 0
            rsi = 100 - (100 / (1 + rs))
            if rsi > 70:
                self.pattern_logger.log_pattern(
                    name='rsi_overbought',
                    time_window=f"{self.data.index[self.current_step-rsi_period]} - {self.data.index[self.current_step-1]}",
                    confidence=float(rsi/100),
                    example_candles=self.data.iloc[self.current_step-rsi_period:self.current_step].reset_index().to_dict('records')
                )
            elif rsi < 30:
                self.pattern_logger.log_pattern(
                    name='rsi_oversold',
                    time_window=f"{self.data.index[self.current_step-rsi_period]} - {self.data.index[self.current_step-1]}",
                    confidence=float((100-rsi)/100),
                    example_candles=self.data.iloc[self.current_step-rsi_period:self.current_step].reset_index().to_dict('records')
                )

    def close(self):
        self.pattern_logger.close()
