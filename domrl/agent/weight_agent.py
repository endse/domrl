"""
Weight Agent — Hybrid SAC-NSGA-II Architecture

Paper Section IV-E: Combines adaptive fine-tuning of deep RL (SAC)
with the global exploration of evolutionary algorithms (NSGA-II).

NSGA-II acts as a 'pre-optimizer' for the policy space, generating
diverse candidate weight vectors. The SAC-based WeightAgent then
refines these through real-time gradient-based optimization.

Paper IV-C: Multi-Objective Reinforcement Learning (MORL)
    Maximize F(x) = {f1(x), -f2(x), f3(x), f4(x), f5(x)}
    where objectives are Engagement, Churn(min), Trust, Diversity, Satisfaction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from domrl.models.networks import StateEncoder
from domrl.agent.nsga2 import NSGA2Optimizer

def quantile_huber_loss(preds, target, tau, kappa=1.0):
    """
    preds: (B, N_QUANTILES)
    target: (B, N_QUANTILES)
    tau: (N_QUANTILES,) - quantile midpoints
    """
    preds = preds.unsqueeze(-1)
    target = target.unsqueeze(1)
    
    u = target - preds
    abs_u = torch.abs(u)
    huber_loss = torch.where(
        abs_u <= kappa,
        0.5 * u.pow(2),
        kappa * (abs_u - 0.5 * kappa)
    )
    
    u_neg = (u < 0).float().detach()
    tau = tau.view(1, -1, 1)
    
    loss = torch.abs(tau - u_neg) * huber_loss
    return loss.mean()

class WeightNetwork(nn.Module):
    """
    Paper IV-E: Actor network for weight generation.
    Outputs 5 gated weights for multi-objective scalarization.
    
    Paper IV-C Table:
    - Engagement (CTR / Watch Time)
    - User Trust (Retention Rate)
    - Churn Mitigation (Inter-session Interval)
    - Diversity (Coverage Score)
    - Satisfaction (Long-term)
    """
    def __init__(self, action_dim=10, num_items=10, hidden_dim=64, num_objectives=5):
        super(WeightNetwork, self).__init__()
        self.num_objectives = num_objectives
        self.encoder = StateEncoder(action_dim=action_dim, num_items=num_items, hidden_dim=hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # Two heads
        self.fc_weights = nn.Linear(hidden_dim, num_objectives)  # Softmax weights
        self.fc_gates = nn.Linear(hidden_dim, num_objectives)    # Sigmoid gates
        
    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.ln1(self.fc1(x)))
        
        raw_weights = F.softmax(self.fc_weights(x), dim=-1)
        gates = torch.sigmoid(self.fc_gates(x))
        
        # Gated Multi-Objective weights
        final_weights = raw_weights * gates
        
        # Scale to maintain consistent reward magnitude
        final_weights = final_weights * float(self.num_objectives)
        
        return final_weights

class QuantileMetaCritic(nn.Module):
    """
    Distributional critic for risk-aware weight optimization.
    Uses Quantile Regression for distributional value estimation.
    """
    def __init__(self, action_dim=10, num_items=10, hidden_dim=64, num_quantiles=25):
        super(QuantileMetaCritic, self).__init__()
        self.num_quantiles = num_quantiles
        self.encoder = StateEncoder(action_dim=action_dim, num_items=num_items, hidden_dim=hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_quantiles)
        
    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.ln1(self.fc1(x)))
        qs = self.fc2(x)
        return qs

class WeightAgent:
    """
    Paper IV-E: Hybrid SAC-NSGA-II Weight Agent.
    
    A shared critic learns to generalize across various goal weightings,
    while a population of actors is evolved using NSGA-II for coverage.
    
    The NSGA-II optimizer periodically evolves the weight population,
    and the best Pareto-front solutions seed the gradient-based
    WeightNetwork training.
    """
    def __init__(self, action_dim=10, num_items=10, hidden_dim=64, lr=1e-4, 
                 gamma=0.99, num_quantiles=25, risk_level=0.1, monotony_weight=1.0,
                 num_objectives=5, nsga2_pop_size=50, nsga2_generations=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.num_quantiles = num_quantiles
        self.risk_level = risk_level
        self.monotony_weight = monotony_weight
        self.num_objectives = num_objectives
        
        # Quantile Midpoints (Constant)
        self.tau = (torch.arange(num_quantiles, device=self.device, dtype=torch.float32) + 0.5) / num_quantiles
        
        self.actor = WeightNetwork(
            action_dim=action_dim, num_items=num_items, 
            hidden_dim=hidden_dim, num_objectives=num_objectives
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = QuantileMetaCritic(
            action_dim=action_dim, num_items=num_items, 
            hidden_dim=hidden_dim, num_quantiles=num_quantiles
        ).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Paper IV-D & IV-E: NSGA-II Optimizer for Pareto-optimal weight discovery
        self.nsga2 = NSGA2Optimizer(
            num_objectives=num_objectives,
            pop_size=nsga2_pop_size,
            num_generations=nsga2_generations,
        )
        
        # Track episode reward vectors for NSGA-II evaluation
        self.episode_reward_vectors = []
        self.nsga2_best_weights = None
        
    def select_weights(self, state):
        """Select objective weights using the WeightNetwork."""
        state_tensor = {}
        for k, v in state.items():
            if k == 'history' or k == 'persona_id':
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
        """
        sorted_qs, _ = torch.sort(quantiles, dim=-1)
        k = int(self.num_quantiles * alpha)
        if k < 1: k = 1
        cvar = sorted_qs[:, :k].mean(dim=1)
        return cvar

    def record_reward_vector(self, reward_vec: np.ndarray):
        """
        Record a reward vector from an episode step for NSGA-II evaluation.
        """
        self.episode_reward_vectors.append(reward_vec.copy())
    
    def evolve_nsga2(self) -> np.ndarray:
        """
        Paper IV-E: Run NSGA-II evolution step.
        
        Evolves the population of weight vectors using accumulated
        reward vectors as objective evaluations. Returns the best
        balanced weights from the Pareto front.
        
        Returns:
            Pareto-optimal weight vector, or None if not enough data
        """
        if len(self.episode_reward_vectors) < 10:
            return None
        
        # Evaluate each weight vector in the population
        reward_matrix = np.array(self.episode_reward_vectors[-100:])  # Last 100 steps
        
        pop_objectives = np.zeros((self.nsga2.pop_size, self.num_objectives))
        for i, weight_vec in enumerate(self.nsga2.population):
            # Compute weighted scores for each objective
            weighted_rewards = reward_matrix * weight_vec[np.newaxis, :]
            pop_objectives[i] = np.mean(weighted_rewards, axis=0)
        
        self.nsga2.set_objectives(pop_objectives)
        
        # Evolve
        pareto_weights = self.nsga2.evolve()
        
        # Get best balanced solution
        self.nsga2_best_weights = self.nsga2.get_best_weights(strategy="balanced")
        
        # Clear accumulated vectors
        self.episode_reward_vectors = self.episode_reward_vectors[-50:]
        
        return self.nsga2_best_weights
    
    def get_nsga2_hypervolume(self) -> float:
        """Get current NSGA-II Pareto front hypervolume for logging."""
        return self.nsga2.get_hypervolume()

    def update(self, replay_buffer, batch_size=64):
        """
        Paper IV-E: Hybrid update combining gradient-based critic/actor
        with NSGA-II Pareto-optimal seeding.
        """
        state, action, next_state, reward, not_done, meta_reward = replay_buffer.sample(batch_size)
        
        # --- 1. Update Meta-Critic (Distributional) ---
        with torch.no_grad():
            next_weights = self.actor(next_state)
            
            next_state_pred = next_state.copy()
            next_state_pred['weights'] = next_weights
            
            next_model_quantiles = self.critic(next_state_pred)
            
            target_quantiles = meta_reward + (self.gamma * next_model_quantiles * not_done)
            
        current_quantiles = self.critic(state)
        
        critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, self.tau)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # --- 2. Update Weight-Actor ---
        pred_weights = self.actor(state)
        
        state_pred = state.copy()
        state_pred['weights'] = pred_weights
        
        pred_quantiles = self.critic(state_pred)
        
        # CVaR objective (Risk-Aware)
        current_cvar = self.calculate_cvar(pred_quantiles, self.risk_level)
        objective_loss = -current_cvar.mean()
        
        # Monotonicity Penalty
        prev_weights = state['weights']
        monotonicity_loss = F.mse_loss(pred_weights, prev_weights)
        
        # Paper IV-E: NSGA-II seeding loss — pull actor toward Pareto-optimal weights
        nsga2_loss = torch.tensor(0.0, device=self.device)
        if self.nsga2_best_weights is not None:
            target_nsga2 = torch.tensor(
                self.nsga2_best_weights, dtype=torch.float32, device=self.device
            ).unsqueeze(0).expand(pred_weights.shape[0], -1)
            nsga2_loss = F.mse_loss(pred_weights, target_nsga2) * 0.1
        
        actor_loss = objective_loss + (self.monotony_weight * monotonicity_loss) + nsga2_loss
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return critic_loss.item(), actor_loss.item()
