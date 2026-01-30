import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserDynamicsNet(nn.Module):
    """
    Neural Network to simulate user internal state transitions.
    Input: [Action(Cat), Previous_Latent_State, Persona_Embedding]
    Output: [Next_Latent_State, Target_Item_Embedding, Satisfaction_Drift]
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

    def forward(self, action_idx, h_prev, persona_idx):
        # Embeddings
        a_emb = self.action_emb(action_idx)
        p_emb = self.persona_emb(persona_idx)
        
        # GRU Input
        x = torch.cat([a_emb, p_emb], dim=-1)
        
        # Latent Update
        h_next = self.gru(x, h_prev)
        
        # Outputs
        target_logits = self.fc_target(h_next)
        sat_mean = torch.sigmoid(self.fc_sat(h_next))
        eng_params = torch.sigmoid(self.fc_eng(h_next)) # [Click Prob, Norm_Scroll]
        
        return h_next, target_logits, sat_mean, eng_params

class GenerativeUserSimulator:
    def __init__(self, num_categories=10, hidden_dim=32):
        self.num_categories = num_categories
        self.hidden_dim = hidden_dim
        
        # Initialize Neural Dynamics
        self.net = UserDynamicsNet(action_dim=num_categories, hidden_dim=hidden_dim)
        
        # We need to manually initialize weights to something reasonable 
        # so the simulator isn't complete garbage initially.
        # However, for an "Simulated Environment", usually the environment logic is fixed 
        # or we load a pre-trained World Model.
        # Since I am 'creating' the environment, I will use randomized weights 
        # but ensure they are deterministic for reproducibility.
        torch.manual_seed(42)
        
        self.reset_state()
        
    def reset_state(self, persona_id=0):
        self.h = torch.zeros(1, self.hidden_dim)
        self.persona_id = torch.tensor([persona_id], dtype=torch.long)
        self.current_satisfaction = 0.5
        
        # Ornstein-Uhlenbeck params
        self.theta = 0.15
        self.sigma = 0.2
        self.dt = 0.1
        
    def step(self, action_input):
        """
        Evolves user state based on agent action (Slate or Item).
        Args:
            action_input: int (Item ID) or list/tuple (Slate of Item IDs)
        """
        is_slate = isinstance(action_input, (list, tuple, np.ndarray)) and len(action_input) > 0
        
        # If slate, user chooses ONE item to interact with (Choice Model)
        # Based on target_logits
        
        # 1. Get Current Latent State & Preferences BEFORE interaction
        # We need a 'dummy' forward pass or store previous logits.
        # Ideally, we query the Net with "Null" action or previous state.
        # But UserDynamicsNet takes (Action, h_prev).
        # Let's assume the user evaluates the slate using CURRENT state h.
        # But to update h to h_next, we strictly need a chosen action.
        
        # Let's perform the Choice process using 'current' preferences (from h_prev essentially)
        # BUT we need logits.
        # Hack: Pass dummy action 0 to get logits from current H?
        # Or better: separating Readout from Update in Network.
        # Given limitations, let's use the 'Action 0' pass to get logits.
        
        dummy_action = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
             # Peek at preferences
             _, target_logits, _, _ = self.net(dummy_action, self.h, self.persona_id)
        
        probs_all = F.softmax(target_logits, dim=-1).numpy()[0] # (Num_Categories,)
        
        chosen_item = 0
        if is_slate:
            # Multinomial Logit on Slate
            slate = list(action_input)
            
            # Extract scores (logits or probs) for items in slate
            slate_probs_unnorm = np.array([probs_all[i] for i in slate])
            # Add 'No Choice' option? (Skip) - optional. For now force choice.
            
            slate_probs = slate_probs_unnorm / (slate_probs_unnorm.sum() + 1e-9)
            
            # Sample Choice
            chosen_idx = np.random.choice(len(slate), p=slate_probs)
            chosen_item = slate[chosen_idx]
        else:
            chosen_item = int(action_input)
            
        # 2. Update Dynamics with CHOSEN Item
        # Now we actually step the RNN
        action_tensor = torch.tensor([chosen_item], dtype=torch.long)
        
        with torch.no_grad():
            h_next, target_logits_next, sat_mean, eng_params = self.net(action_tensor, self.h, self.persona_id)
            
        self.h = h_next
        
        # --- 3. Satisfaction Dynamics ---
        # Driven by the outcome of the interaction (chosen item)
        target_sat = sat_mean.item()
        noise = np.random.normal(0, np.sqrt(self.dt))
        dx = self.theta * (target_sat - self.current_satisfaction) * self.dt + self.sigma * noise
        self.current_satisfaction = np.clip(self.current_satisfaction + dx, 0.0, 1.0)
        
        # --- 4. Signals ---
        # Did choice match true intent?
        # Re-eval 'True Intent' from logits (what they wanted most from ALL items)
        true_intent = np.argmax(probs_all)
        
        features = eng_params[0].numpy()
        click_prob_base = features[0]
        
        is_match = (chosen_item == true_intent)
        
        if is_match:
             click_prob = min(0.95, click_prob_base + 0.4)
             self.current_satisfaction = min(1.0, self.current_satisfaction + 0.15)
             scroll_val = 0.5
             hover_val = np.random.normal(2.5, 0.5)
        else:
             # If they chose something sub-optimal from slate, minimal satisfaction gain?
             click_prob = max(0.01, click_prob_base - 0.1)
             if is_slate:
                 # If slate didn't contain true intent, frustration
                 if true_intent not in action_input:
                     self.current_satisfaction -= 0.05
             else:
                 self.current_satisfaction -= 0.05
                 
             scroll_val = 5.0
             hover_val = 0.5

        did_click = 1.0 if np.random.rand() < click_prob else 0.0
        view_time = hover_val + (1.0 if did_click else 0.0)
        
        signals = {
            "click": did_click,
            "scroll": scroll_val,
            "hover": hover_val,
            "view_time": view_time,
            "chosen_item": chosen_item
        }
        
        return self.current_satisfaction, signals
