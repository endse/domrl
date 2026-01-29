import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from domrl.models.networks import StateEncoder

class WeightNetwork(nn.Module):
    def __init__(self, action_dim=10, hidden_dim=64):
        super(WeightNetwork, self).__init__()
        self.encoder = StateEncoder(action_dim=action_dim, hidden_dim=hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3) # Output 3 weights
        
    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        weights = F.softmax(x, dim=-1)
        return weights

class MetaCritic(nn.Module):
    def __init__(self, action_dim=10, hidden_dim=64):
        super(MetaCritic, self).__init__()
        # StateEncoder includes weights in the input embedding
        self.encoder = StateEncoder(action_dim=action_dim, hidden_dim=hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Output Scalar Q-Val
        
    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.fc1(x))
        q = self.fc2(x)
        return q

class WeightAgent:
    def __init__(self, action_dim=10, lr=1e-4, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        
        self.actor = WeightNetwork(action_dim=action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = MetaCritic(action_dim=action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
    def select_weights(self, state):
        # Convert state dict to tensor
        state_tensor = {}
        for k, v in state.items():
            if k == 'history':
                state_tensor[k] = torch.LongTensor(v).to(self.device).unsqueeze(0)
            else:
                state_tensor[k] = torch.FloatTensor(v).to(self.device).unsqueeze(0)
                
        with torch.no_grad():
            weights = self.actor(state_tensor)
        return weights.squeeze(0).cpu().numpy()
    
    def update(self, replay_buffer, batch_size=64):
        state, action, next_state, reward, not_done, meta_reward = replay_buffer.sample(batch_size)
        
        # --- 1. Update Meta-Critic ---
        with torch.no_grad():
            # Target = r_meta + gamma * Q(s', w')
            # Note: next_state already contains next_weights taken in the environment?
            # Actually, standard DDPG/SAC uses Target Actor for next action.
            # Here, the weights in next_state are what was actually taken (if any).
            # But we want to estimate max Q? Or Q of current policy?
            # Let's use Target Policy: w_next = actor(next_state)
            
            # Use current actor for next weights (DDPG style, no target net for now for simplicity)
            next_weights = self.actor(next_state)
            
            # Construct next_state with predicted weights for the critic
            next_state_pred = next_state.copy()
            next_state_pred['weights'] = next_weights
            
            target_q = self.critic(next_state_pred)
            target_value = meta_reward + (self.gamma * target_q * not_done)
            
        # Current Q
        current_q = self.critic(state) # state['weights'] are the weights actually used
        
        critic_loss = F.mse_loss(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- 2. Update Weight-Actor ---
        # Maximize Q(s, actor(s))
        pred_weights = self.actor(state)
        
        state_pred = state.copy()
        state_pred['weights'] = pred_weights
        
        actor_loss = -self.critic(state_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return critic_loss.item(), actor_loss.item()

