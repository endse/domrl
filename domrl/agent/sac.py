"""
Soft Actor-Critic (SAC) Agent

Paper Section IV-A & IV-B: Maximum Entropy Reinforcement Learning.

The SAC agent maximizes both cumulative reward and policy entropy:
    J(π) = Σ E[r(s_t, a_t) + α·H(π(·|s_t))]

Key components:
- Twin critics (Q1, Q2) to mitigate overestimation bias
- Entropy-regularized policy with dynamic temperature α
- Soft Bellman equation for value estimation
- Reparameterization trick with tanh squashing correction
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from domrl.models.networks import Actor, Critic

class SACAgent:
    """
    Paper IV-B: Soft Actor-Critic with entropy-augmented exploration.
    
    The temperature parameter α controls exploration vs exploitation:
    - Higher α → broader exploration of state-action manifold
    - Lower α → exploitation of known high-reward trajectories
    """
    def __init__(self, state_dim, action_dim, num_items=10, hidden_dim=256, 
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, auto_alpha=True, 
                 cql_weight=0.0, bc_weight=0.0, grad_clip=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        # Paper IV-A: Target entropy H_target = -log(1/|A|) * 0.98
        self.target_entropy = -np.log(1.0 / action_dim) * 0.98
        self.cql_weight = cql_weight
        self.bc_weight = bc_weight
        self.grad_clip = grad_clip  # Gradient clipping for stability

        # Paper IV-B: Actor generates distribution over content slates
        self.actor = Actor(state_dim, action_dim, num_items=num_items, hidden_dim=hidden_dim).to(self.device)
        # Paper IV-B: Twin critic networks (Q1, Q2) — using min(Q1, Q2) mitigates overestimation
        self.critic = Critic(state_dim, action_dim, num_items=num_items, hidden_dim=hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, num_items=num_items, hidden_dim=hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Paper IV-A: Dynamic temperature parameter α
        if self.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
    def select_action(self, state, evaluate=False):
        """
        Select action using the current policy.
        
        Paper IV-B: In evaluation mode, uses deterministic action (mean).
        In training, samples from the stochastic policy for exploration.
        """
        # State is a dict of numpy arrays. Convert to dict of tensors with batch dim.
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
            if evaluate:
                mean, log_std = self.actor(state_tensor)
                action = torch.tanh(mean) # Deterministic action for eval
            else:
                action, _, _ = self.actor.sample(state_tensor)
                
        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        """
        Paper IV-B: Full SAC update with:
        1. Soft Bellman equation for critic update
        2. Entropy-regularized actor update
        3. Dynamic alpha tuning
        4. Polyak averaging for target network
        """
        state, action, next_state, reward, not_done, _ = replay_buffer.sample(batch_size)
        
        # ------------------- #
        #  Critic Update      #
        #  Paper IV-B: Soft Bellman Equation
        #  Q(s,a) = r(s,a) + γ·E[Q(s',a') - α·log π(a'|s')]
        # ------------------- #
        with torch.no_grad():
            # Sample next action from current policy
            next_action, next_log_probs, _ = self.actor.sample(next_state)
            
            # Twin critics: use min(Q1, Q2) to prevent overestimation
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next)
            
            # Soft Value: V(s') = Q(s',a') - α·log_π(a'|s')
            target_q = reward + not_done * self.gamma * (q_next - self.alpha * next_log_probs)
        
        # Current Q estimates
        q1, q2 = self.critic(state, action)
        
        # Standard SAC Critic Loss
        sac_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        critic_loss = sac_loss 
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        
        # ------------------- #
        #  Actor Update       #
        #  Paper IV-A: Maximize E[Q - α·log_π]
        #  Loss: α·log_π - Q
        # ------------------- #
        new_action, log_probs, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (self.alpha * log_probs - q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        
        # ------------------- #
        #  Alpha Update       #
        #  Paper IV-A: Dynamic temperature tuning
        #  α_loss = -α·(log_π + H_target)
        # ------------------- #
        entropy = -log_probs.mean().item()
        
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()

        # ------------------- #
        #  Polyak Update      #
        #  Soft target update: θ_target ← τ·θ + (1-τ)·θ_target
        # ------------------- #
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return critic_loss.item(), actor_loss.item(), self.alpha, list(q1.detach().cpu().numpy()), entropy
