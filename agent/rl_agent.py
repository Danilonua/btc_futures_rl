import logging
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Any, Dict

logger = logging.getLogger(__name__)

ALGOS = {
    'ppo': PPO,
    'dqn': DQN,
    'sac': SAC,
}

class RLAgent:
    """
    RL Agent wrapper for stable-baselines3 algorithms.
    Supports PPO, DQN, SAC (extendable).
    """
    def __init__(self, env, config: Dict[str, Any]):
        self.env = DummyVecEnv([lambda: env])
        self.config = config
        self.algo_name = config['agent']['algo'].lower()
        self.model = None
        self._init_model()

    def _init_model(self):
        algo_cls = ALGOS.get(self.algo_name)
        if not algo_cls:
            logger.error(f"Unsupported algorithm: {self.algo_name}")
            raise ValueError(f"Unsupported algorithm: {self.algo_name}")
        try:
            self.model = algo_cls(
                policy=self.config['agent'].get('policy', 'MlpPolicy'),
                env=self.env,
                learning_rate=self.config['agent']['learning_rate'],
                batch_size=self.config['agent']['batch_size'],
                gamma=self.config['agent']['gamma'],
                n_steps=self.config['agent'].get('n_steps', 2048),
                ent_coef=self.config['agent'].get('ent_coef', 0.01),
                verbose=1
            )
            logger.info(f"Initialized {self.algo_name.upper()} agent.")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise

    def train(self, total_timesteps: int):
        try:
            logger.info(f"Training for {total_timesteps} timesteps...")
            self.model.learn(total_timesteps=total_timesteps)
            logger.info("Training complete.")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def save(self, path: str):
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load(self, path: str):
        try:
            self.model = ALGOS[self.algo_name].load(path, env=self.env)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
