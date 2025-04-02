import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from collections import deque

class PolicyMemory:
    def __init__(self, decay_rate=0.9):
        self.memory = deque(maxlen=1000)
        self.decay_rate = decay_rate

    def add_policy(self, date, policy_type, value, outcomes, expected_effects=None):
        entry = {
            'date': date,
            'type': policy_type,
            'value': value,
            'outcomes': outcomes,
            'expected': expected_effects,
            'efficacy': self._calculate_efficacy(outcomes, expected_effects)
        }
        self.memory.append(entry)

    def _calculate_efficacy(self, outcomes, expected):
        if not expected:
            return 0.5
        total_error = sum(abs(outcomes.get(key, 0) - expected.get(key, 0)) for key in outcomes)
        return np.exp(-total_error)

    def get_weighted_history(self, current_date):
        features = {
            'monetary': {'value': 0, 'efficacy': 0, 'recency': 0},
            'fiscal': {'value': 0, 'efficacy': 0, 'recency': 0}
        }
        
        for policy in self.memory:
            days_passed = current_date - policy['date']
            weight = self.decay_rate ** days_passed
            
            if policy['type'] in features:
                f = features[policy['type']]
                f['value'] += policy['value'] * weight
                f['efficacy'] += policy['efficacy'] * weight
                f['recency'] += weight
        
        for key in features:
            if features[key]['recency'] > 0:
                features[key]['value'] /= features[key]['recency']
                features[key]['efficacy'] /= features[key]['recency']
        
        return features

class PolicyAwareEconomy(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self):
        super().__init__()
        
        # Action space: [interest_rate, fiscal_stimulus]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]), 
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space as Box for compatibility
        self.observation_space = spaces.Box(
            low=np.concatenate([np.array([-10]*4), np.array([0]*6)]),
            high=np.concatenate([np.array([10]*4), np.array([1]*6)]),
            dtype=np.float32
        )
        
        self.policy_memory = PolicyMemory()
        self.reset()

    def step(self, action):
        interest_rate, fiscal_stimulus = action
        gdp, inflation, unemployment, debt = self.state[:4]
        
        # Economic dynamics
        gdp += 0.1 * fiscal_stimulus - 0.2 * interest_rate + 0.05 * fiscal_stimulus * interest_rate
        inflation += 0.3 * interest_rate - 0.1 * fiscal_stimulus
        unemployment -= 0.2 * gdp
        
        # Store policy outcomes only if we have a previous policy to compare with
        if self.last_policy is not None:  # Only store if we have a previous policy
            outcomes = {
                'delta_gdp': gdp - self.state[0],
                'delta_inflation': inflation - self.state[1]
            }
            self.policy_memory.add_policy(
                date=self.current_date,
                policy_type='monetary',
                value=self.last_policy[0],
                outcomes=outcomes
            )
        
        # Update state
        self.state = np.concatenate([
            np.array([gdp, inflation, unemployment, debt], dtype=np.float32),
            self._get_policy_features()
        ])
        self.current_date += 1
        self.last_policy = action  # Store current action for next step
        
        reward = - (inflation**2 + unemployment**2 + 0.1*debt**2)
        terminated = False
        truncated = False
        info = {}
        
        return self.state.copy(), reward, terminated, truncated, info

    def _get_policy_features(self):
        history = self.policy_memory.get_weighted_history(self.current_date)
        return np.array([
            history['monetary']['value'],
            history['monetary']['efficacy'],
            history['monetary']['recency'],
            history['fiscal']['value'],
            history['fiscal']['efficacy'],
            history['fiscal']['recency']
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.state = np.concatenate([
            np.array([1.0, 0.02, 0.05, 0.6], dtype=np.float32),
            np.zeros(6, dtype=np.float32)
        ])
        self.current_date = 0
        self.last_policy = None
        
        # Gymnasium requires returning info dict
        info = {}
        return self.state.copy(), info

    def render(self):
        print(f"Month: {self.current_date}")
        print(f"GDP: {self.state[0]:.2f}, Inflation: {self.state[1]:.2f}")
        print(f"Unemployment: {self.state[2]:.2f}, Debt: {self.state[3]:.2f}")

if __name__ == "__main__":
    # Create and wrap environment
    env = DummyVecEnv([lambda: Monitor(PolicyAwareEconomy())])
    eval_env = DummyVecEnv([lambda: Monitor(PolicyAwareEconomy())])
    
    # Configure callbacks
    callbacks = [
        EvalCallback(eval_env, 
                    best_model_save_path='./models/',
                    log_path='./logs/', 
                    eval_freq=1000,
                    deterministic=True, 
                    render=False)
    ]
    
    # Model Training
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard/",
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=10,
        max_grad_norm=0.5
    )
    
    print("Training model...")
    model.learn(
        total_timesteps=50000,
        callback=callbacks,
        tb_log_name="ppo_econ_policy",
        progress_bar=False
    )
    
    # Save the model
    model.save("econ_policy_agent")
    print("Model saved successfully!")