import pytest
import numpy as np
from env.btc_futures_env import BTCFuturesEnv
import pandas as pd

@pytest.fixture
def sample_config():
    return {
        'fee': 0.0004,
        'slippage': 0.0002,
        'max_position': 1.0,
        'max_leverage': 2,
        'stop_loss': 0.02,
        'take_profit': 0.04,
        'var_window': 10,
        'var_quantile': 0.01,
        'initial_balance': 10000,
        'symbol': 'BTC/USDT',
        'timeframe': '1m',
    }

@pytest.fixture
def sample_data():
    # 100 steps of fake OHLCV data
    np.random.seed(42)
    data = np.random.rand(100, 5) * 20000 + 20000
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
    return df

def test_env_reset_and_step(sample_config, sample_data):
    env = BTCFuturesEnv(sample_config, sample_data)
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)
    obs, reward, done, info = env.step(1)  # Long
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def test_reward_function(sample_config, sample_data):
    env = BTCFuturesEnv(sample_config, sample_data)
    env.reset()
    # Open long, then close
    obs, _, _, _ = env.step(1)
    obs, reward, done, info = env.step(3)
    # Reward should be net profit minus fees/slippage
    assert isinstance(reward, float)
    # Should not crash on repeated closes
    obs, reward, done, info = env.step(3)
    assert isinstance(reward, float)
