import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from domrl.models.networks import Actor, Critic

class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, auto_alpha=True, cql_weight=0.0, bc_weight=0.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.target_entropy = -np.log(1.0 / action_dim) * 0.98
        self.cql_weight = cql_weight
        self.bc_weight = bc_weight

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        if self.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
    def select_action(self, state, evaluate=False):
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
                _, probs, _ = self.actor.sample(state_tensor)
                action = torch.argmax(probs, dim=1)
            else:
                action, _, _ = self.actor.sample(state_tensor)
        return action.item()

    def update(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done, _ = replay_buffer.sample(batch_size)
        
        # ------------------- #
        #  Critic Update      #
        # ------------------- #
        with torch.no_grad():
            # Get probabilities for next state
            _, next_probs, next_log_probs = self.actor.sample(next_state)
            
            # Get Q-values for next state from target critic
            q1_next, q2_next = self.critic_target(next_state)
            q_next = torch.min(q1_next, q2_next)
            
            # Soft Value: V(s') = E_a [ Q(s',a) - alpha * log_pi(a|s') ]
            # Sum over all actions: sum(probs * (q - alpha * log_probs))
            v_next = torch.sum(next_probs * (q_next - self.alpha * next_log_probs), dim=1, keepdim=True)
            
            target_q = reward + not_done * self.gamma * v_next
        
        # Current Q estimates
        q1, q2 = self.critic(state) # (B, ActionDim)
        
        # Standard SAC Loss
        q1_taken = q1.gather(1, action.view(-1, 1))
        q2_taken = q2.gather(1, action.view(-1, 1))
        sac_loss = F.mse_loss(q1_taken, target_q) + F.mse_loss(q2_taken, target_q)
        
        # CQL Loss: log(sum(exp(Q))) - Q(s, a)
        if self.cql_weight > 0:
            cql_loss1 = torch.logsumexp(q1, dim=1).mean() - q1_taken.mean()
            cql_loss2 = torch.logsumexp(q2, dim=1).mean() - q2_taken.mean()
            critic_loss = sac_loss + self.cql_weight * (cql_loss1 + cql_loss2)
        else:
            critic_loss = sac_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ------------------- #
        #  Actor Update       #
        # ------------------- #
        _, probs, log_probs = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state)
        q_pi = torch.min(q1_pi, q2_pi)
        
        # Objective: Maximize E[Q - alpha * log_pi]
        # Loss: Minimize -sum(probs * (Q - alpha * log_probs))
        sac_actor_loss = torch.sum(probs * (self.alpha * log_probs - q_pi), dim=1).mean()
        
        # Behavior Cloning Loss (Supervised)
        # Maximize log_prob(action_taken)
        # Minimize -log_prob(action_taken)
        if self.bc_weight > 0:
            # We need log_probs of the *taken* action, not sampled
            # probs: (B, A)
            dist = torch.distributions.Categorical(probs)
            log_prob_taken = dist.log_prob(action.view(-1))
            bc_loss = -log_prob_taken.mean()
            
            actor_loss = sac_actor_loss + self.bc_weight * bc_loss
        else:
            actor_loss = sac_actor_loss
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ------------------- #
        #  Alpha Update       #
        # ------------------- #
        if self.auto_alpha:
            # Alpha Loss: -alpha * (log_pi + target_entropy)
            # Using current probs directly without re-sampling for efficiency (approx)
            alpha_loss = torch.sum(probs.detach() * (-self.log_alpha * (log_probs.detach() + self.target_entropy)), dim=1).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()

        # ------------------- #
        #  Polyak Update      #
        # ------------------- #
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return critic_loss.item(), actor_loss.item(), self.alpha, list(q1.detach().cpu().numpy())
