import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RealTimeRecEnv(gym.Env):
    """
    DOM-RL Environment (MOMDP) with User Personas.
    State: 
        - History: Sequence of last N items interacted with.
        - UserFeatures: [Enthusiasm, Time].
        - MicroSignals: [ScrollVelocity, HoverDuration, ViewTime].
        - Weights: [w_eng, w_sat, w_div].
        - Persona: One-hot or ID of the user persona.
    Action: Discrete item category (0-9).
    Reward: Vector or Composite.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(RealTimeRecEnv, self).__init__()
        
        self.history_len = 10
        self.num_categories = 10
        
        self.personas = ["Standard", "Binger", "Browser", "Critic"]
        self.num_personas = len(self.personas)
        
        # Observation Space as Dict
        self.observation_space = spaces.Dict({
            "history": spaces.Box(0, self.num_categories, shape=(self.history_len,), dtype=np.int32),
            "user_features": spaces.Box(0, 24, shape=(2,), dtype=np.float32), # [Enthusiasm, Time]
            "micro_signals": spaces.Box(-10, 60, shape=(3,), dtype=np.float32), # [Scroll, Hover, View]
            "weights": spaces.Box(0, 5, shape=(3,), dtype=np.float32), # [w1, w2, w3]
            "persona_id": spaces.Discrete(self.num_personas) # [0, 1, 2, 3]
        })

        self.action_space = spaces.Discrete(self.num_categories)
        self.max_steps = 100
        
        # Internal State
        self.current_step = 0
        self.user_satisfaction = 0.5
        self.history = np.zeros(self.history_len, dtype=np.int32)
        self.current_persona_id = 0
        
    def _get_obs(self) -> dict:
        """
        Returns the current observation dictionary.
        
        Returns:
            dict: Dictionary containing:
                - 'history': np.ndarray (int32) of shape (10,)
                - 'user_features': np.ndarray (float32) of shape (2,)
                - 'micro_signals': np.ndarray (float32) of shape (3,)
                - 'weights': np.ndarray (float32) of shape (3,)
                - 'persona_id': int
        """
        return {
            "history": self.history.copy(),
            "user_features": self.user_state.astype(np.float32),
            "micro_signals": self.micro_signals.astype(np.float32),
            "weights": self.weights.astype(np.float32),
            "persona_id": self.current_persona_id
        }

    def reset(self, seed: int = None, options: dict = None) -> tuple[dict, dict]:
        """
        Resets the environment to an initial state.
        
        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Configuration dictionary. 
                - 'weights': list[float] (Overwrites random weights)
                - 'persona_id': int (Forces a specific persona)
        
        Returns:
            tuple[dict, dict]: Observation and info dictionary.
        """
        super().reset(seed=seed)
        self.current_step = 0
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
            
        # Select Persona
        if options and 'persona_id' in options:
            self.current_persona_id = options['persona_id']
        else:
            self.current_persona_id = np.random.randint(0, self.num_personas)
            
        # Initial User State (Persona-dependent)
        base_enthusiasm = np.random.rand()
        if self.current_persona_id == 1: # Binger
            base_enthusiasm += 0.3
            self.user_satisfaction = 0.7
        else:
            self.user_satisfaction = 0.5
            
        self.user_state = np.array([
            min(1.0, base_enthusiasm),
            np.random.rand() * 24   # Time
        ], dtype=np.float32)
        
        self.micro_signals = np.array([0.0, 0.0, 0.0], dtype=np.float32) # Scroll, Hover, View
        
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[dict, np.ndarray, bool, bool, dict]:
        """
        Executes one time step within the environment.
        
        Args:
            action (int): The category ID of the recommended item.
            
        Returns:
            tuple:
                - observation (dict): Next state.
                - reward (np.ndarray): Multi-objective reward vector [Eng, Sat, Div].
                - terminated (bool): Whether the episode ended (Churn).
                - truncated (bool): Whether the episode timed out (MaxSteps).
                - info (dict): Diagnostic info.
        """
        self.current_step += 1
        
        # Update History (Shift left, append new)
        self.history = np.roll(self.history, -1)
        self.history[-1] = action
        
        # --- Using Persona Logic ---
        persona = self.personas[self.current_persona_id]
        
        # User Preference Logic
        preferred_category = int(self.user_state[1]) % self.num_categories
        is_match = (action == preferred_category)
        
        decay = 0.05
        boost = 0.1
        
        # Persona Modifiers
        if persona == "Binger":
            decay = 0.02 # More tolerant
            boost = 0.15 # Easily satisfied
        elif persona == "Critic":
            decay = 0.1  # Harsh penalty
            boost = 0.05 # Hard to please
            # Critic only likes it if visualization is perfect (simulated by extra random check)
            if is_match and np.random.rand() < 0.3:
                is_match = False # Nitpicking
                
        if is_match:
            self.user_satisfaction = min(1.0, self.user_satisfaction + boost)
            hover = np.random.normal(2.0, 0.5)
            scroll = 0.5
            click = 1
        else:
            self.user_satisfaction = max(0.0, self.user_satisfaction - decay)
            hover = 0.1
            scroll = 5.0
            click = 0
            
        # Browser persona scrolls faster
        if persona == "Browser":
            scroll += 3.0
            hover *= 0.5
            
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
        
        # Churn threshold depends on persona?
        churn_thresh = 0.2
        if persona == "Binger": churn_thresh = 0.1
        if persona == "Critic": churn_thresh = 0.3
        
        if self.user_satisfaction < churn_thresh:
            if np.random.rand() < 0.3:
                terminated = True
                r_satisfaction -= 10.0 # Heavy penalty for churn
        
        if self.current_step >= self.max_steps:
             terminated = True
             
        reward_vector = np.array([r_engage, r_satisfaction, r_diversity], dtype=np.float32)
             
        return self._get_obs(), reward_vector, terminated, truncated, {}
