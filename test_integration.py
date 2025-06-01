from env.btc_futures_env import BTCFuturesEnv
import numpy as np
import pandas as pd

class DummyAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []

    def act(self, state):
        return np.random.randint(self.action_size)

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def replay(self, batch_size):
        pass

# Тестовые данные
np.random.seed(42)
raw_data = pd.DataFrame({
    'open': np.random.rand(1000) * 10000 + 20000,
    'high': np.random.rand(1000) * 10000 + 20000,
    'low': np.random.rand(1000) * 10000 + 20000,
    'close': np.random.rand(1000) * 10000 + 20000,
    'volume': np.random.rand(1000) * 10 + 1
})
config = {
    'fee': 0.0004, 'slippage': 0.0002, 'max_position': 1.0, 'max_leverage': 2,
    'stop_loss': 0.02, 'take_profit': 0.04, 'var_window': 10, 'var_quantile': 0.01,
    'initial_balance': 10000, 'symbol': 'BTC/USDT', 'timeframe': '1m'
}
# Добавим индикаторы вручную, чтобы избежать ошибки размерности
from env.btc_futures_env import BTCFuturesEnv
processed_data = BTCFuturesEnv._add_indicators(None, raw_data)
env = BTCFuturesEnv(config, processed_data)
state, _ = env.reset()
agent = DummyAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

for step in range(100):
    action = agent.act(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    agent.remember(state, action, reward, next_state, terminated or truncated)
    if len(agent.memory) > 32:
        agent.replay(32)
    state = next_state
    if terminated or truncated:
        break

print("✔ Интеграционный тест пройден!")
print(f"Финальный баланс: ${env.equity:.2f}")
