import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from domrl.env.user_simulator import GenerativeUserSimulator
from domrl.utils.slate_utils import SlateMapper

class RealTimeRecEnv(gym.Env):
    """
    DOM-RL Environment (MOMDP) with User Personas and Slate Recommendations.
    
    Paper III: Real-time adaptive system with granular micro-behavior monitoring.
    Paper III-C: Multi-Objective Optimization with 5 objectives:
        1. Engagement (CTR / Watch Time)
        2. User Satisfaction (Retention Rate)
        3. Diversity (Coverage Score)
        4. Fairness (Persona Satisfaction Gap)
        5. Churn Mitigation (Inter-session Interval)
    
    Paper III (Challenge C): Cold Start behavioral inference.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, slate_size=3):
        super(RealTimeRecEnv, self).__init__()
        
        self.history_len = 10
        self.num_categories = 10
        self.slate_size = slate_size
        self.num_objectives = 5  # Paper III-C: 5 objectives
        
        self.personas = ["Standard", "Binger", "Browser", "Critic"]
        self.num_personas = len(self.personas)
        
        # Continuous Action Space: Embedding Vector (Dim=16)
        self.embedding_dim = 16
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.embedding_dim,), dtype=np.float32)
        
        # Simulator
        self.simulator = GenerativeUserSimulator(num_categories=self.num_categories)
        
        # Paper III-A: Expanded Observation Space
        # micro_signals now has 6 dimensions:
        #   [scroll_velocity, hover_dwell_ratio, skip_gradient, scroll, hover, view_time]
        self.observation_space = spaces.Dict({
            "history": spaces.Box(0, self.num_categories, shape=(self.history_len,), dtype=np.int32),
            "user_features": spaces.Box(0, 24, shape=(2,), dtype=np.float32), 
            "micro_signals": spaces.Box(-10, 60, shape=(6,), dtype=np.float32),  # Expanded: 3 -> 6
            "weights": spaces.Box(0, 5, shape=(self.num_objectives,), dtype=np.float32),  # 4 -> 5
            "persona_id": spaces.Discrete(self.num_personas) 
        })
        
        self.max_steps = 100
        
        # Internal State
        self.current_step = 0
        self.user_satisfaction = 0.5
        self.history = np.zeros(self.history_len, dtype=np.int32)
        self.current_persona_id = 0
        
        # Fairness Tracking
        self.action_counts = np.ones(self.num_categories) 
        self.persona_satisfaction = {p: 0.5 for p in range(self.num_personas)} 
        
        # Paper III-C: Churn tracking
        self.consecutive_low_sat_steps = 0
        self.churn_probability = 0.0
        
        # Cold Start (Paper III Challenge C)
        self.cold_start_phase = True
        self.cold_start_steps = 5
        
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
        self.consecutive_low_sat_steps = 0
        self.churn_probability = 0.0
        self.cold_start_phase = True
        
        # Dynamic Weights (5 objectives now)
        if options and 'weights' in options:
             self.weights = np.array(options['weights'], dtype=np.float32)
        else:
            self.weights = np.array([
                np.random.uniform(0.5, 1.5),  # Engagement
                np.random.uniform(0.1, 1.0),  # Satisfaction
                np.random.uniform(1.0, 3.0),  # Diversity
                np.random.uniform(0.5, 2.0),  # Fairness
                np.random.uniform(0.5, 1.5),  # Churn Mitigation
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
        
        # Paper III-A: Expanded micro-signals (6 dimensions)
        self.micro_signals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Ensure Item Matrix is built
        from domrl.utils.movie_db import get_movie_db
        from domrl.config import cfg
        self.db = get_movie_db(cfg.MOVIE_LENS_PATH)
        if self.db.item_embeddings is None:
             self.db._build_item_embeddings()
        
        return self._get_obs(), {}

    def set_user_context(self, mood, time_of_day):
        """Pass context to simulator"""
        self.simulator.set_context(mood, time_of_day)
        self.user_state[1] = time_of_day

    def _infer_cold_start_persona(self):
        """
        Paper III (Challenge C): The Dynamic Cold Start Problem.
        Infer persona from navigation style within the first few seconds
        before any click or rating occurs.
        """
        inferred = self.simulator.infer_cold_start_persona()
        if inferred != self.current_persona_id:
            self.current_persona_id = inferred
            self.simulator.persona_id = torch.tensor([inferred], dtype=torch.long)
        self.cold_start_phase = False

    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray, bool, bool, dict]:
        """
        Executes one time step. 
        Action is Continuous Embedding (Shape: 16,)
        
        Paper III-A: Processes enriched micro-behavioral signals.
        Paper III-C: Computes 5-objective reward vector.
        """
        self.current_step += 1
        
        # Cold Start Phase Check (Paper III Challenge C)
        if self.cold_start_phase and self.current_step >= self.cold_start_steps:
            self._infer_cold_start_persona()
        
        # 1. Map Action Embedding to Slate via ANN
        results = self.db.search_nearest_items(action, k=self.slate_size)
        slate_mids = [r[0] for r in results]
        
        # Convert MIDs to Categories for Simulator/History
        slate_cats = [self.db.movie_cat_map.get(mid, 0) for mid in slate_mids]
        
        # 2. Simulator Step with Slate
        sat, signals = self.simulator.step(slate_cats)
        self.user_satisfaction = sat
        
        chosen_cat = signals.get('chosen_item', slate_cats[0])
        chosen_idx_in_slate = 0
        try:
            chosen_idx_in_slate = slate_cats.index(chosen_cat)
        except ValueError:
            pass
            
        chosen_mid = slate_mids[chosen_idx_in_slate]
        chosen_title = results[chosen_idx_in_slate][1]
        
        # 3. Update History
        self.history = np.roll(self.history, -1)
        self.history[-1] = chosen_cat
        
        # Parse Enriched Signals (Paper III-A)
        scroll_velocity = signals.get('scroll_velocity', signals['scroll'])
        hover_dwell_ratio = signals.get('hover_dwell_ratio', 0.5)
        skip_gradient = signals.get('skip_gradient', 0.0)
        scroll = signals['scroll']
        hover = signals['hover']
        view_time = signals['view_time']
        click = signals['click']
        
        # Update Micro-signals (6 dimensions)
        self.micro_signals = np.array([
            scroll_velocity,      # Paper III-A: Scroll velocity analysis
            hover_dwell_ratio,    # Paper III-A: Hover-dwell micro-dynamics
            skip_gradient,        # Paper III-A: Skip-rate temporal gradient
            scroll,               # Legacy: raw scroll
            hover,                # Legacy: raw hover
            view_time,            # Legacy: view time
        ], dtype=np.float32)
        
        # --- Update Fairness Stats ---
        self.action_counts[chosen_cat] += 1
        
        alpha = 0.1
        self.persona_satisfaction[self.current_persona_id] = (
            (1 - alpha) * self.persona_satisfaction[self.current_persona_id] + alpha * sat
        )
        
        # --- Paper III-C: Churn Probability Tracking ---
        if sat < 0.3:
            self.consecutive_low_sat_steps += 1
        else:
            self.consecutive_low_sat_steps = max(0, self.consecutive_low_sat_steps - 1)
        
        # Churn probability increases with consecutive low-satisfaction steps
        self.churn_probability = min(1.0, self.consecutive_low_sat_steps * 0.1)
        
        # --- 5-Objective Reward Calculation (Paper III-C) ---
        # 1. Engagement: (Click=1, else 0) * (ViewTime/Max)
        r_eng = float(click) + (view_time / 60.0)
        
        # 2. Satisfaction: Survey result (0-1) -> scaled
        r_sat = sat * 2.0 - 1.0 
        
        # 3. Diversity: Unique categories in slate / slate_size
        unique_items = len(set(slate_cats))
        r_div = (unique_items / self.slate_size) * 2.0 - 1.0 
        
        # 4. Fairness: Persona Sat Gap
        sats = list(self.persona_satisfaction.values())
        r_fair = 1.0 - (max(sats) - min(sats))
        
        # 5. Churn Mitigation (Paper III-C): Inverse probability of session termination
        r_churn = 1.0 - self.churn_probability
        
        reward_vec = np.array([r_eng, r_sat, r_div, r_fair, r_churn], dtype=np.float32)
        
        # Scaled Reward using Weights (5-dim dot product)
        scalar_reward = np.dot(reward_vec, self.weights)
        
        done = self.current_step >= self.max_steps
        truncated = False
        
        # Churn Logic
        persona = self.personas[self.current_persona_id]
        churn_thresh = 0.2
        if persona == "Binger": churn_thresh = 0.1
        if persona == "Critic": churn_thresh = 0.3
        
        if self.user_satisfaction < churn_thresh:
            if np.random.rand() < 0.3:
                done = True
                scalar_reward -= 5.0
        
        info = {
            "signals": signals,
            "weights": self.weights,
            "chosen_cat": chosen_cat,
            "chosen_title": chosen_title,
            "slate": [r[1] for r in results],
            "reward_vector": reward_vec,
            "churn_probability": self.churn_probability,
            "cold_start": signals.get('is_cold_start', False),
        }
        
        return self._get_obs(), scalar_reward, done, truncated, info
