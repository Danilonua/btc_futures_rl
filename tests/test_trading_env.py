import unittest
import numpy as np
import pandas as pd
from env.trading_env import TradingEnv


class TestTradingEnv(unittest.TestCase):
    def setUp(self):
        data = pd.DataFrame({
            'open': np.linspace(100, 200, 100),
            'high': np.linspace(105, 205, 100),
            'low': np.linspace(95, 195, 100),
            'close': np.linspace(102, 202, 100),
            'volume': np.random.randint(100, 1000, 100)
        })
        config = {
            'env': {
                'window_size': 10,
                'initial_balance': 10000,
                'max_position': 0.1,
                'commission': 0.0004,
                'stop_loss': 0.03,
                'take_profit': 0.06
            }
        }
        self.env = TradingEnv(data, config)

    def test_reset(self):
        obs = self.env.reset()
        self.assertEqual(obs.shape, (10, 7))

    def test_step(self):
        self.env.reset()
        action = np.array([1, 0, 0.5])  # Buy 50%
        obs, reward, done, _ = self.env.step(action)
        self.assertIsInstance(reward, float)


if __name__ == '__main__':
    unittest.main()