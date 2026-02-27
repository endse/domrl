import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class UserDynamicsNet(nn.Module):
    """
    Neural Network to simulate user internal state transitions.
    Input: [Action(Cat), Previous_Latent_State, Persona_Embedding]
    Output: [Next_Latent_State, Target_Item_Embedding, Satisfaction_Drift]
    
    Paper Section III-A: Implicit Signal Architecture and Feature Engineering
    """
    def __init__(self, action_dim=10, hidden_dim=32, persona_dim=4):
        super(UserDynamicsNet, self).__init__()
        
        self.action_emb = nn.Embedding(action_dim, 8)
        self.persona_emb = nn.Embedding(4, persona_dim)
        
        # GRU for Latent State of the User (Mood/Intent)
        self.gru = nn.GRUCell(input_size=8 + persona_dim, hidden_size=hidden_dim)
        
        # Readout heads
        # 1. Target Item Logits : What the user ideally wants next
        self.fc_target = nn.Linear(hidden_dim, action_dim)
        
        # 2. Satisfaction Mean: The 'center' of satisfaction for the current state
        self.fc_sat = nn.Linear(hidden_dim, 1) 
        
        # 3. Engagement Params: Click Probability, Scroll Speed
        self.fc_eng = nn.Linear(hidden_dim, 2) 

        # 4. Micro-Behavioral Signal Heads (Paper III-A)
        # Scroll velocity (continuous, higher = high-entropy search)
        self.fc_scroll_velocity = nn.Linear(hidden_dim, 1)
        # Hover-dwell dynamics (ratio of dwell on card vs session)
        self.fc_hover_dwell = nn.Linear(hidden_dim, 1)
        # Skip-rate temporal gradient (skip timing signal)
        self.fc_skip_gradient = nn.Linear(hidden_dim, 1)

    def forward(self, action_idx, h_prev, persona_idx):
        """
        Legacy forward for compatibility, but performs one full step:
        h_prev -> [Preferences] -> (Implicit Selection) -> Action -> h_next
        But for training we just want: (Action, h_prev) -> h_next, targets
        """
        # This acts as the TRANSITION function: f(h, a) -> h'
        return self.forward_transition(action_idx, h_prev, persona_idx)

    def forward_transition(self, action_idx, h_prev, persona_idx):
        # Embeddings
        a_emb = self.action_emb(action_idx)
        p_emb = self.persona_emb(persona_idx)
        
        # GRU Input
        x = torch.cat([a_emb, p_emb], dim=-1)
        
        # Latent Update
        h_next = self.gru(x, h_prev)
        
        target_logits = self.fc_target(h_next)
        sat_mean = torch.sigmoid(self.fc_sat(h_next))
        eng_params = torch.sigmoid(self.fc_eng(h_next))
        
        return h_next, target_logits, sat_mean, eng_params

    def get_preferences(self, h):
        """
        Get preferences (logits) for the CURRENT state h.
        Used for Choice Model.
        """
        return self.fc_target(h)
    
    def get_aux_signals(self, h):
        sat_mean = torch.sigmoid(self.fc_sat(h))
        eng_params = torch.sigmoid(self.fc_eng(h))
        return sat_mean, eng_params

    def get_micro_behavioral_signals(self, h):
        """
        Paper III-A: Extract granular micro-behavioral signals from latent state.
        Returns scroll_velocity, hover_dwell_ratio, skip_gradient
        """
        scroll_vel = torch.sigmoid(self.fc_scroll_velocity(h)) * 10.0  # 0-10 range
        hover_dwell = torch.sigmoid(self.fc_hover_dwell(h))  # 0-1 ratio
        skip_grad = torch.tanh(self.fc_skip_gradient(h))  # -1 to 1 gradient
        return scroll_vel, hover_dwell, skip_grad


class GenerativeUserSimulator:
    """
    Simulates a user interacting with a recommendation system.
    
    Paper III-A: Implements Granular Feature Engineering (GFE) —
    scroll velocity analysis, hover-dwell micro-dynamics, 
    skip-rate temporal gradients.
    """
    def __init__(self, num_categories=10, hidden_dim=32):
        self.num_categories = num_categories
        self.hidden_dim = hidden_dim
        
        # Initialize Neural Dynamics
        self.net = UserDynamicsNet(action_dim=num_categories, hidden_dim=hidden_dim)
        
        # Load Pre-trained weights if available
        checkpoint_path = "domrl/checkpoints/user_model_v2.pth"
        if os.path.exists(checkpoint_path):
            try:
                self.net.load_state_dict(torch.load(checkpoint_path))
                print(f"GenerativeUserSimulator loaded World Model from {checkpoint_path}")
            except Exception as e:
                print(f"Failed to load World Model: {e}")
        else:
            print("GenerativeUserSimulator using Random Weights (Untrained)")
        
        torch.manual_seed(42)
        
        self.reset_state()
        
    def reset_state(self, persona_id=0):
        self.h = torch.zeros(1, self.hidden_dim)
        self.persona_id = torch.tensor([persona_id], dtype=torch.long)
        self.current_satisfaction = 0.5
        
        # Ornstein-Uhlenbeck params (Restored)
        self.theta = 0.15
        self.sigma = 0.2
        self.dt = 0.1
        
        # Context Variables
        self.current_mood = 0 # 0: Neutral, 1: Happy, 2: Sad, 3: Tired
        self.time_of_day = 12.0 # 0-24
        
        # --- Paper III-A: Micro-Behavioral Tracking State ---
        self.session_duration = 0.0       # Total session time in seconds
        self.sim_history = []             # Internal interaction history
        self.skip_history = []            # (timestamp_seconds, item_id) for skip analysis
        self.interaction_timestamps = []  # Timestamps of all interactions
        self.hover_durations = []         # Hover durations per step
        self.scroll_velocities = []       # Scroll velocity per step
        
        # Cold Start (Paper III: Challenge C)
        self.cold_start_interactions = 0
        self.cold_start_scroll_pattern = []   # Scroll velocities during cold start
        self.cold_start_hover_pattern = []    # Hover durations during cold start
        
    def set_context(self, mood, time_of_day):
        self.current_mood = mood
        self.time_of_day = time_of_day
        
    def _apply_context_bias(self, logits):
        """
        Adjusts logits based on Mood and Time of Day.
        Logits shape: (Num_Categories,) - numpy array or tensor
        """
        # Mappings (Indices based on MovieDatabase default map)
        # 0:Action/Adv, 1:Comedy/Kids, 2:Drama/Rom, 3:SciFi/Fant, 4:Crime/Thril, 5:Horror, 6:Doc, 7:Music, 8:West, 9:Noir
        
        bias = np.zeros_like(logits)
        
        # 1. MOOD BIAS
        if self.current_mood == 1: # Happy
            bias[1] += 2.0  # Comedy
            bias[0] += 1.0  # Adventure
            bias[7] += 1.0  # Musical
        elif self.current_mood == 2: # Sad
            bias[1] += 1.5  # Comedy (Cheer up)
            bias[2] += 2.0  # Drama (Wallow)
        elif self.current_mood == 3: # Tired
            bias[6] += 2.0  # Documentary (Passive)
            bias[1] += 1.0  # Kids (Simple)
            bias[4] -= 1.0  # Avoid Thrillers (Too intense)
            
        # 2. TIME BIAS
        # Night (20:00 - 04:00) -> Horror, Thriller
        if self.time_of_day >= 20 or self.time_of_day < 4:
            bias[5] += 2.5 # Horror
            bias[4] += 1.5 # Thriller/Crime
            bias[9] += 1.0 # Noir
        # Morning (06:00 - 11:00) -> Avoid Horror, prefer Doc/Kids
        elif 6 <= self.time_of_day < 11:
            bias[5] -= 3.0 # No Horror in morning
            bias[6] += 1.0 # Doc
            bias[1] += 0.5 # Animation
            
        return logits + bias

    def _compute_scroll_velocity(self, is_match, click_prob):
        """
        Paper III-A: Scroll Velocity Analysis.
        High velocity = high-entropy search (browsing rapidly).
        Rapid deceleration = focusing on content clusters.
        """
        with torch.no_grad():
            base_vel = self.net.get_micro_behavioral_signals(self.h)[0].item()
        
        if is_match:
            # User found what they want — deceleration (low velocity)
            velocity = max(0.1, base_vel * 0.3 + np.random.normal(0, 0.2))
        else:
            # User not satisfied — high-entropy search (high velocity)
            velocity = min(10.0, base_vel * 1.5 + np.random.normal(0, 0.5))
        
        self.scroll_velocities.append(velocity)
        return velocity

    def _compute_hover_dwell(self, is_match, chosen_item):
        """
        Paper III-A: Hover-Dwell Micro-Dynamics.
        Ratio of dwell time on content card vs total session duration.
        Differentiates active engagement from passive idle.
        """
        with torch.no_grad():
            base_dwell = self.net.get_micro_behavioral_signals(self.h)[1].item()
        
        if is_match:
            # Active engagement — longer dwell time
            dwell_time = max(0.5, base_dwell * 5.0 + np.random.normal(2.5, 0.5))
        else:
            # Passive/dismissive — short dwell
            dwell_time = max(0.1, base_dwell * 1.0 + np.random.normal(0.5, 0.3))
        
        self.hover_durations.append(dwell_time)
        
        # Calculate ratio against session duration
        if self.session_duration > 0:
            hover_dwell_ratio = min(1.0, dwell_time / max(1.0, self.session_duration))
        else:
            hover_dwell_ratio = 0.5  # Neutral for first interaction
        
        return dwell_time, hover_dwell_ratio

    def _compute_skip_gradient(self, chosen_item, did_click, view_time):
        """
        Paper III-A: Skip-Rate Temporal Gradients.
        Skip < 3s = failure in visual/titular 'grab' (early_skip_signal).
        Skip > 12s = lack of sustained content quality (late_skip_signal).
        Not a binary negative — gradient based on timing.
        """
        SKIP_SHORT_THRESHOLD = 3.0   # seconds
        SKIP_LONG_THRESHOLD = 12.0   # seconds
        
        if did_click > 0.5:
            # User clicked — no skip. Positive gradient.
            skip_gradient = 0.5
        else:
            # User skipped — analyze timing
            self.skip_history.append((self.session_duration, chosen_item))
            
            if view_time < SKIP_SHORT_THRESHOLD:
                # Early skip: titular/visual failure
                skip_gradient = -1.0
            elif view_time < SKIP_LONG_THRESHOLD:
                # Mid skip: partial engagement then left
                skip_gradient = -0.3
            else:
                # Late skip (>12s): content quality issue, not visual
                skip_gradient = -0.1
        
        return skip_gradient

    def infer_cold_start_persona(self):
        """
        Paper III (Challenge C): The Dynamic Cold Start Problem.
        Infer user persona within the first few seconds from navigation style
        before any click or rating.
        
        Returns inferred persona_id based on scroll/hover patterns.
        """
        if len(self.cold_start_scroll_pattern) < 2:
            return 0  # Default: Standard
        
        avg_scroll = np.mean(self.cold_start_scroll_pattern)
        avg_hover = np.mean(self.cold_start_hover_pattern) if self.cold_start_hover_pattern else 1.0
        scroll_variance = np.var(self.cold_start_scroll_pattern)
        
        # Persona inference rules based on navigation behavior:
        # Binger: fast scroll, low hover (rapid consumption)
        # Browser: medium scroll, medium hover (exploring)
        # Critic: slow scroll, high hover (careful evaluation)
        # Standard: default
        
        if avg_scroll > 5.0 and avg_hover < 1.5:
            return 1  # Binger
        elif avg_scroll < 2.0 and avg_hover > 2.5:
            return 3  # Critic
        elif scroll_variance > 3.0:
            return 2  # Browser (variable patterns)
        else:
            return 0  # Standard
        
    def step(self, action_input):
        """
        Evolves user state based on agent action (Slate or Item).
        Args:
            action_input: int (Item ID) or list/tuple (Slate of Item IDs)
            
        Paper III-A: Returns enriched micro-behavioral signals including
        scroll velocity, hover-dwell ratio, and skip-rate temporal gradients.
        """
        is_slate = isinstance(action_input, (list, tuple, np.ndarray)) and len(action_input) > 0
        
        # Track session time (simulated)
        step_duration = np.random.uniform(2.0, 15.0)  # seconds per step
        self.session_duration += step_duration
        self.cold_start_interactions += 1
        
        # 1. Get Current Latent State & Preferences BEFORE interaction
        with torch.no_grad():
             target_logits = self.net.get_preferences(self.h)
        
        logits_np = target_logits.numpy()[0]
        logits_biased = self._apply_context_bias(logits_np)
        
        probs_all = F.softmax(torch.tensor(logits_biased), dim=-1).numpy()
        
        chosen_item = 0
        if is_slate:
            slate = list(action_input)
            slate_probs_unnorm = np.array([probs_all[i] for i in slate])
            slate_probs = slate_probs_unnorm / (slate_probs_unnorm.sum() + 1e-9)
            chosen_idx = np.random.choice(len(slate), p=slate_probs)
            chosen_item = slate[chosen_idx]
        else:
            chosen_item = int(action_input)
            
        # 2. Update Dynamics with CHOSEN Item
        action_tensor = torch.tensor([chosen_item], dtype=torch.long)
        
        with torch.no_grad():
            h_next, target_logits_next, sat_mean, eng_params = self.net.forward_transition(action_tensor, self.h, self.persona_id)
            
        self.h = h_next
        
        # --- 3. Satisfaction Dynamics ---
        target_sat = sat_mean.item()
        
        # BOREDOM / REPETITIVENESS PENALTY
        self.sim_history.append(chosen_item)
        if len(self.sim_history) > 10: self.sim_history.pop(0)
        
        repetition_count = self.sim_history.count(chosen_item)
        boredom_penalty = 0.0
        if repetition_count > 2:
            boredom_penalty = 0.1 * (repetition_count - 2)
            
        target_sat = max(0.0, target_sat - boredom_penalty)

        # DIVERSITY BONUS (Novelty)
        diversity_bonus = 0.0
        if is_slate:
            unique_slate = len(set(action_input))
            if unique_slate == len(action_input):
                 diversity_bonus = 0.05
        
        target_sat = min(1.0, target_sat + diversity_bonus)

        noise = np.random.normal(0, np.sqrt(self.dt))
        dx = self.theta * (target_sat - self.current_satisfaction) * self.dt + self.sigma * noise
        self.current_satisfaction = np.clip(self.current_satisfaction + dx, 0.0, 1.0)
        
        # --- 4. Enriched Micro-Behavioral Signals (Paper III-A) ---
        true_intent = np.argmax(probs_all)
        
        features = eng_params[0].numpy()
        click_prob_base = features[0]
        
        is_match = (chosen_item == true_intent)
        
        if is_match:
             click_prob = min(0.95, click_prob_base + 0.4)
             sat_gain = 0.15 if repetition_count <= 2 else 0.05
             self.current_satisfaction = min(1.0, self.current_satisfaction + sat_gain)
        else:
             click_prob = max(0.01, click_prob_base - 0.1)
             if is_slate:
                 if true_intent not in action_input:
                     self.current_satisfaction -= 0.05
             else:
                 self.current_satisfaction -= 0.05

        did_click = 1.0 if np.random.rand() < click_prob else 0.0
        
        # Paper III-A: Compute enriched signals
        scroll_velocity = self._compute_scroll_velocity(is_match, click_prob)
        hover_time, hover_dwell_ratio = self._compute_hover_dwell(is_match, chosen_item)
        
        view_time = hover_time + (3.0 if did_click else 0.0)
        skip_gradient = self._compute_skip_gradient(chosen_item, did_click, view_time)
        
        # Track cold start patterns (Paper III Challenge C)
        if self.cold_start_interactions <= 5:
            self.cold_start_scroll_pattern.append(scroll_velocity)
            self.cold_start_hover_pattern.append(hover_time)
        
        signals = {
            "click": did_click,
            # Paper III-A: Scroll Velocity Analysis
            "scroll": scroll_velocity,
            "scroll_velocity": scroll_velocity,
            # Paper III-A: Hover-Dwell Micro-Dynamics
            "hover": hover_time,
            "hover_dwell_ratio": hover_dwell_ratio,
            # Paper III-A: Skip-Rate Temporal Gradients
            "view_time": view_time,
            "skip_gradient": skip_gradient,
            # Metadata
            "chosen_item": chosen_item,
            "session_duration": self.session_duration,
            "is_cold_start": self.cold_start_interactions <= 5,
        }
        
        return self.current_satisfaction, signals
