import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from domrl.models.networks import StateEncoder

def quantile_huber_loss(preds, target, tau, kappa=1.0):
    """
    preds: (B, N_QUANTILES)
    target: (B, N_QUANTILES)
    tau: (N_QUANTILES,) - quantile midpoints
    """
    # preds: (B, N, 1)
    # target: (B, 1, N)
    preds = preds.unsqueeze(-1)
    target = target.unsqueeze(1)
    
    # huber loss
    u = target - preds # (B, N, N)
    abs_u = torch.abs(u)
    huber_loss = torch.where(
        abs_u <= kappa,
        0.5 * u.pow(2),
        kappa * (abs_u - 0.5 * kappa)
    )
    
    # quantile regression loss
    # delta(u < 0) = 1 if u < 0 else 0
    # loss = |tau - delta(u<0)| * huber_loss
    u_neg = (u < 0).float().detach()
    tau = tau.view(1, -1, 1) # Broadcast to (1, N, 1) for the 'prediction' dimension
    
    # Note: QR-DQN usually does:
    # element-wise loss between every pred quantile and every target quantile
    # Then sum/mean.
    
    loss = torch.abs(tau - u_neg) * huber_loss
    return loss.mean()

class WeightNetwork(nn.Module):
    def __init__(self, action_dim=10, hidden_dim=64):
        super(WeightNetwork, self).__init__()
        self.encoder = StateEncoder(action_dim=action_dim, hidden_dim=hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 4) # Output 4 weights (Eng, Sat, Div, Fair)
        
    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        weights = F.softmax(x, dim=-1)
        return weights

class QuantileMetaCritic(nn.Module):
    def __init__(self, action_dim=10, hidden_dim=64, num_quantiles=25):
        super(QuantileMetaCritic, self).__init__()
        self.num_quantiles = num_quantiles
        # StateEncoder includes weights in the input embedding
        self.encoder = StateEncoder(action_dim=action_dim, hidden_dim=hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_quantiles) # Output N quantiles
        
    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.fc1(x))
        qs = self.fc2(x)
        return qs

class WeightAgent:
    def __init__(self, action_dim=10, hidden_dim=64, lr=1e-4, gamma=0.99, num_quantiles=25, risk_level=0.1, monotony_weight=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.num_quantiles = num_quantiles
        self.risk_level = risk_level # Alpha for CVaR (e.g., 0.1 for worst 10%)
        self.monotony_weight = monotony_weight
        
        # Quantile Midpoints (Constant)
        self.tau = (torch.arange(num_quantiles, device=self.device, dtype=torch.float32) + 0.5) / num_quantiles
        
        self.actor = WeightNetwork(action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = QuantileMetaCritic(action_dim=action_dim, hidden_dim=hidden_dim, num_quantiles=num_quantiles).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
    def select_weights(self, state):
        # Convert state dict to tensor
        state_tensor = {}
        for k, v in state.items():
            if k == 'history' or k == 'persona_id':
                # Ensure correct shape (1, Sequence) or (1,)
                t = torch.as_tensor(v, device=self.device, dtype=torch.long)
                if t.dim() == 0: t = t.unsqueeze(0)
                if t.dim() == 1 and k=='history': t = t.unsqueeze(0)
                state_tensor[k] = t
            else:
                t = torch.as_tensor(v, device=self.device, dtype=torch.float32)
                if t.dim() == 1: t = t.unsqueeze(0)
                state_tensor[k] = t
                
        with torch.no_grad():
            weights = self.actor(state_tensor)
        return weights.squeeze(0).cpu().numpy()
    
    def calculate_cvar(self, quantiles, alpha=0.1):
        """
        Calculate conditional value at risk for the lower tail (worst cases).
        Since quantiles are outputted (approx), we take the lowest ones up to alpha.
        """
        # Quantiles are largely unordered coming out of NN unless we enforce it, 
        # but for QR-DQN we usually sort them for interpretation or C51.
        # Actually QR-DQN minimizes Wasserstein distance to sorted targets, so they tend to sort.
        # Let's sort explicitly.
        sorted_qs, _ = torch.sort(quantiles, dim=-1)
        
        # Index for the cutoff
        # e.g. 25 quantiles. alpha=0.1 => use 2.5 => 3 quantiles.
        k = int(self.num_quantiles * alpha)
        if k < 1: k = 1
        
        # Average the worst k outcomes
        cvar = sorted_qs[:, :k].mean(dim=1)
        return cvar

    def update(self, replay_buffer, batch_size=64):
        state, action, next_state, reward, not_done, meta_reward = replay_buffer.sample(batch_size)
        
        # --- 1. Update Meta-Critic (Distributional) ---
        with torch.no_grad():
            # Target Policy: w_next = actor(next_state)
            next_weights = self.actor(next_state)
            
            # Predict Quantiles at next state
            next_state_pred = next_state.copy()
            next_state_pred['weights'] = next_weights
            
            next_model_quantiles = self.critic(next_state_pred) # (B, N)
            
            # Bellman Update for Quantiles: r + gamma * Z(s')
            # meta_reward is (B, 1), mask is (B, 1)
            target_quantiles = meta_reward + (self.gamma * next_model_quantiles * not_done) # Broadcased
            
        # Current Prediction
        current_quantiles = self.critic(state) # (B, N)
        
        # Quantile Huber Loss
        critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, self.tau)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- 2. Update Weight-Actor ---
        # Objective: Maximize CVaR of Z(s, actor(s))
        # Contraint: Minimize || w_new - w_old ||^2 (Monotonicity/Stability)
        
        pred_weights = self.actor(state)
        
        state_pred = state.copy()
        state_pred['weights'] = pred_weights
        
        pred_quantiles = self.critic(state_pred)
        
        # Calculate CVaR (Risk-Aware Objective)
        # We want to maximize the lower tail (worst outcomes)
        current_cvar = self.calculate_cvar(pred_quantiles, self.risk_level)
        objective_loss = -current_cvar.mean()
        
        # Monotonicity Penalty
        # current weights in state are 'prev_weights', pred_weights are 'new_weights'
        prev_weights = state['weights']
        monotonicity_loss = F.mse_loss(pred_weights, prev_weights)
        
        actor_loss = objective_loss + (self.monotony_weight * monotonicity_loss)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return critic_loss.item(), actor_loss.item()
