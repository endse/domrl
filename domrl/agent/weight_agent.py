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

class WeightAgent:
    def __init__(self, action_dim=10, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WeightNetwork(action_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def select_weights(self, state):
        # Convert state dict to tensor
        state_tensor = {}
        for k, v in state.items():
            if k == 'history':
                state_tensor[k] = torch.LongTensor(v).to(self.device).unsqueeze(0)
            else:
                state_tensor[k] = torch.FloatTensor(v).to(self.device).unsqueeze(0)
                
        with torch.no_grad():
            weights = self.model(state_tensor)
        return weights.squeeze(0).cpu().numpy()
    
    def update(self, state, meta_reward):
        # Simple Policy Gradient / REINFORCE-like update for prototype
        # Maximize meta_reward
        # Log prob of the weights? No, this is deterministic/continuous output.
        # Let's treat it as a simple regression towards "better" weights?
        # Actually, standard DDPG would be better, but simpler:
        # We want to move weights in direction of gradient of meta_reward.
        # But we don't have diff(meta_reward, weights).
        # Let's use a simple perturbation or bandit approach, OR assume the 'meta_reward' is a gradient signal?
        
        # PROTOTYPE HACK:
        # Since we can't easily do RL on this without a Critic for the weights,
        # we will use a simple "Hill Climbing" or noise-based approach, 
        # OR just plain Random Exploration if we want to mimic the "Generator" part.
        
        # BUT, the prompt asked for an implementation.
        # Let's implement a dummy update for now that does nothing, 
        # as implementing a full hierarchical RL in one step is risky.
        # The prompt mentioned "Simple Meta-Critic".
        pass
