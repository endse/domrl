import gymnasium as gym
from gymnasium import spaces
import numpy as np
from domrl.env.user_simulator import GenerativeUserSimulator
from domrl.utils.slate_utils import SlateMapper

class RealTimeRecEnv(gym.Env):
    """
    DOM-RL Environment (MOMDP) with User Personas and Slate Recommendations.
    State: 
        - History: Sequence of last N items interacted with.
        - ...
    Action: Discrete index mapping to a Slate of items.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, slate_size=3):
        super(RealTimeRecEnv, self).__init__()
        
        self.history_len = 10
        self.num_categories = 10
        self.slate_size = slate_size
        
        self.personas = ["Standard", "Binger", "Browser", "Critic"]
        self.num_personas = len(self.personas)
        
        # Slate Mapper
        self.slate_mapper = SlateMapper(num_items=self.num_categories, slate_size=slate_size, allow_repeats=True)
        self.num_actions = self.slate_mapper.num_actions
        
        # Simulator
        self.simulator = GenerativeUserSimulator(num_categories=self.num_categories)
        
        # Observation Space as Dict
        self.observation_space = spaces.Dict({
            "history": spaces.Box(0, self.num_categories, shape=(self.history_len,), dtype=np.int32),
            "user_features": spaces.Box(0, 24, shape=(2,), dtype=np.float32), 
            "micro_signals": spaces.Box(-10, 60, shape=(3,), dtype=np.float32), 
            "weights": spaces.Box(0, 5, shape=(4,), dtype=np.float32), 
            "persona_id": spaces.Discrete(self.num_personas) 
        })

        self.action_space = spaces.Discrete(self.num_actions)
        self.max_steps = 100
        
        # Internal State
        self.current_step = 0
        self.user_satisfaction = 0.5
        self.history = np.zeros(self.history_len, dtype=np.int32)
        self.current_persona_id = 0
        
        # Fairness Tracking
        self.action_counts = np.ones(self.num_categories) 
        self.persona_satisfaction = {p: 0.5 for p in range(self.num_personas)} 
        
    def _get_obs(self) -> dict:
        return {
            "history": self.history.copy(),
            "user_features": self.user_state.astype(np.float32),
            "micro_signals": self.micro_signals.astype(np.float32),
            "weights": self.weights.astype(np.float32),
            "persona_id": self.current_persona_id
        }

    def reset(self, seed: int = None, options: dict = None) -> tuple[dict, dict]:
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
                np.random.uniform(1.0, 3.0),  # Diversity
                np.random.uniform(0.5, 2.0)   # Fairness
            ], dtype=np.float32)
            
        # Select Persona
        if options and 'persona_id' in options:
            self.current_persona_id = options['persona_id']
        else:
            self.current_persona_id = np.random.randint(0, self.num_personas)
            
        # Reset Simulator with current Persona
        self.simulator.reset_state(persona_id=self.current_persona_id)
        self.user_satisfaction = self.simulator.current_satisfaction
            
        # Initial User State (Persona-dependent)
        base_enthusiasm = np.random.rand()
        self.user_state = np.array([
            min(1.0, base_enthusiasm),
            np.random.rand() * 24   # Time
        ], dtype=np.float32)
        
        self.micro_signals = np.array([0.0, 0.0, 0.0], dtype=np.float32) 
        
        return self._get_obs(), {}

    def set_user_context(self, mood, time_of_day):
        """Pass context to simulator"""
        self.simulator.set_context(mood, time_of_day)
        # Optionally update user_features[1] with normalized time for the Agent to see
        # time is 0-24. Normalize to 0-1? Or just raw.
        # user_features is shape (2,). [0] is enthusiasm. [1] was random "Time" (0-24).
        # Let's sync it.
        self.user_state[1] = time_of_day

    def step(self, action: int) -> tuple[dict, np.ndarray, bool, bool, dict]:
        """
        Executes one time step. Action is Slate Index.
        """
        self.current_step += 1
        
        # 1. Map Action Index to Slate Tuple
        slate = self.slate_mapper.get_slate(action) # e.g. (1, 5, 5)
        
        # 2. Simulator Step with Slate
        sat, signals = self.simulator.step(slate)
        self.user_satisfaction = sat
        
        chosen_item = signals.get('chosen_item', slate[0])
        
        # 3. Update History with CHOSEN item
        self.history = np.roll(self.history, -1)
        self.history[-1] = chosen_item
        
        # Parse Signals
        scroll = signals['scroll']
        hover = signals['hover']
        view_time = signals['view_time']
        click = signals['click']
        
        # Update Micro-signals
        self.micro_signals = np.array([scroll, hover, view_time], dtype=np.float32)
        
        # --- Update Fairness Stats ---
        self.action_counts[chosen_item] += 1
        
        alpha = 0.1
        self.persona_satisfaction[self.current_persona_id] = (1 - alpha) * self.persona_satisfaction[self.current_persona_id] + alpha * sat
        
        # --- Reward Calculation ---
        if len(self.weights) < 4:
            self.weights = np.pad(self.weights, (0, 4 - len(self.weights)), mode='constant', constant_values=1.0)
            
        w_eng, w_sat, w_div, w_fair = self.weights
        
        r_engage = click * 1.0
        r_satisfaction = self.user_satisfaction
        
        # Diversity / Submodularity Reward on Slate
        # Penalize if slate contains duplicates (redundancy)
        unique_items = len(set(slate))
        redundancy_penalty = (len(slate) - unique_items) * 0.5 
        r_diversity = (unique_items / len(slate)) - redundancy_penalty
        
        # Fairness Reward Calculation on Chosen Item
        total_counts = self.action_counts.sum()
        prob = self.action_counts[chosen_item] / total_counts
        r_exposure = 0.5 / prob if prob > 0 else 1.0 
        r_exposure = min(r_exposure, 5.0) 
        
        global_avg_sat = np.mean(list(self.persona_satisfaction.values()))
        persona_avg_sat = self.persona_satisfaction[self.current_persona_id]
        
        r_demo = 0.0
        if persona_avg_sat < global_avg_sat - 0.1: 
            r_demo = 1.0 
            
        r_fairness = 0.5 * r_exposure + 0.5 * r_demo
        
        # Churn Logic
        terminated = False
        truncated = False
        
        persona = self.personas[self.current_persona_id]
        churn_thresh = 0.2
        if persona == "Binger": churn_thresh = 0.1
        if persona == "Critic": churn_thresh = 0.3
        
        if self.user_satisfaction < churn_thresh:
            if np.random.rand() < 0.3:
                terminated = True
                r_satisfaction -= 10.0 
        
        if self.current_step >= self.max_steps:
             terminated = True
             
        reward_vector = np.array([r_engage, r_satisfaction, r_diversity, r_fairness], dtype=np.float32)
             
        return self._get_obs(), reward_vector, terminated, truncated, {}
