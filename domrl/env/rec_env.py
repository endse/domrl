import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RealTimeRecEnv(gym.Env):
    """
    DOM-RL Environment (MOMDP)
    State: 
        - History: Sequence of last N items interacted with.
        - UserFeatures: [Enthusiasm, Time].
        - MicroSignals: [ScrollVelocity, HoverDuration, ViewTime].
        - Weights: [w_eng, w_sat, w_div].
    Action: Discrete item category (0-9).
    Reward: Vector or Composite.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(RealTimeRecEnv, self).__init__()
        
        self.history_len = 10
        self.num_categories = 10
        
        # Observation Space as Dict
        self.observation_space = spaces.Dict({
            "history": spaces.Box(0, self.num_categories, shape=(self.history_len,), dtype=np.int32),
            "user_features": spaces.Box(0, 24, shape=(2,), dtype=np.float32), # [Enthusiasm, Time]
            "micro_signals": spaces.Box(-10, 60, shape=(3,), dtype=np.float32), # [Scroll, Hover, View]
            "weights": spaces.Box(0, 5, shape=(3,), dtype=np.float32) # [w1, w2, w3]
        })

        self.action_space = spaces.Discrete(self.num_categories)
        self.max_steps = 100
        
        # Internal State
        self.current_step = 0
        self.user_satisfaction = 0.5
        self.history = np.zeros(self.history_len, dtype=np.int32)
        
    def _get_obs(self):
        return {
            "history": self.history.copy(),
            "user_features": self.user_state.astype(np.float32),
            "micro_signals": self.micro_signals.astype(np.float32),
            "weights": self.weights.astype(np.float32)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.user_satisfaction = 0.5
        self.history = np.zeros(self.history_len, dtype=np.int32)
        
        # Dynamic Weights
        if options and 'weights' in options:
             self.weights = np.array(options['weights'], dtype=np.float32)
        else:
            self.weights = np.array([
                np.random.uniform(0.5, 1.5),  # Engagement
                np.random.uniform(0.1, 1.0),  # Satisfaction
                np.random.uniform(1.0, 3.0)   # Diversity/Churn
            ], dtype=np.float32)
            
        # Initial User State
        self.user_state = np.array([
            np.random.rand(),       # Enthusiasm
            np.random.rand() * 24   # Time
        ], dtype=np.float32)
        
        self.micro_signals = np.array([0.0, 0.0, 0.0], dtype=np.float32) # Scroll, Hover, View
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # Update History (Shift left, append new)
        self.history = np.roll(self.history, -1)
        self.history[-1] = action
        
        # --- Simulation Logic ---
        # User Preference Logic
        preferred_category = int(self.user_state[1]) % self.num_categories
        is_match = (action == preferred_category)
        
        if is_match:
            self.user_satisfaction = min(1.0, self.user_satisfaction + 0.1)
            hover = np.random.normal(2.0, 0.5)
            scroll = 0.5
            click = 1
        else:
            self.user_satisfaction = max(0.0, self.user_satisfaction - 0.05)
            hover = 0.1
            scroll = 5.0
            click = 0
            
        view_time = hover + (0.5 if click else 0.0)
        
        # Update Micro-signals
        self.micro_signals = np.array([scroll, hover, view_time], dtype=np.float32)
        
        # --- Reward Calculation ---
        w_eng, w_sat, w_div = self.weights
        
        r_engage = click * 1.0
        r_satisfaction = self.user_satisfaction
        
        # Diversity placeholder (Compare action to recent history)
        # Simple diversity: 1 if action not in last 3 items, else 0
        recent = self.history[-4:-1] # excluding current which is at -1
        r_diversity = 1.0 if action not in recent else 0.0
        
        # Muti-Objective Reward Vector
        # [Engagement, Satisfaction, Diversity]
        
        # Churn Logic
        terminated = False
        truncated = False
        
        if self.user_satisfaction < 0.2:
            if np.random.rand() < 0.3:
                terminated = True
                r_satisfaction -= 10.0 # Heavy penalty for churn
        
        if self.current_step >= self.max_steps:
             terminated = True
             
        reward_vector = np.array([r_engage, r_satisfaction, r_diversity], dtype=np.float32)
             
        return self._get_obs(), reward_vector, terminated, truncated, {}
