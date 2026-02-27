import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class StateEncoder(nn.Module):
    """
    Encodes the dictionary state into a fixed-size embedding.
    
    Paper III-A: Now processes 6 micro-behavioral signals
    (scroll_velocity, hover_dwell_ratio, skip_gradient + legacy scroll, hover, view_time).
    """
    def __init__(self, action_dim=10, num_items=10, hidden_dim=64, micro_dim=6):
        super(StateEncoder, self).__init__()
        
        self.micro_dim = micro_dim
        
        # History Encoder
        # Item IDs 0-(num_items-1). Embedding needs num_items + 1 (for padding)
        self.embedding = nn.Embedding(num_embeddings=num_items + 1, embedding_dim=32)
        self.gru = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        
        # Persona Encoder
        self.persona_embedding = nn.Embedding(num_embeddings=4, embedding_dim=4)
        
        # Combined Feature Dimension
        # GRU_Out(64) + User(2) + Micro(micro_dim) + Weights(5) + Persona(4)
        combined_dim = 64 + 2 + micro_dim + 5 + 4
        self.fc = nn.Linear(combined_dim, hidden_dim)
        # Paper IV-B: LayerNorm for training stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, state_dict):
        # 1. History -> GRU
        h = state_dict['history'] # (B, 10)
        emb = self.embedding(h)   # (B, 10, 32)
        _, h_n = self.gru(emb)    # h_n: (1, B, 64)
        h_seq = h_n.squeeze(0)    # (B, 64)
        
        # 2. Other features
        u = state_dict['user_features']     # (B, 2)
        m = state_dict['micro_signals']     # (B, 6)
        w = state_dict['weights']           # (B, 5)
        p = state_dict['persona_id']        # (B,)
        
        if p.dim() == 2 and p.shape[1] == 1:
             p = p.squeeze(1)
        
        p_emb = self.persona_embedding(p) # (B, 4)
        
        # 3. Concatenate
        cat = torch.cat([h_seq, u, m, w, p_emb], dim=1)
        
        # 4. Dense Encode with LayerNorm
        x = F.relu(self.layer_norm(self.fc(cat)))
        return x

class Critic(nn.Module):
    """
    Twin Q-Networks (Q1, Q2) for SAC.
    
    Paper IV-B: Twin critics mitigate overestimation bias.
    Uses LayerNorm between hidden layers for stability.
    """
    def __init__(self, state_dim, action_dim, num_items=10, hidden_dim=256, micro_dim=6):
        super(Critic, self).__init__()
        self.encoder = StateEncoder(action_dim=0, num_items=num_items, hidden_dim=hidden_dim, micro_dim=micro_dim)
        
        # Q1 architecture with LayerNorm
        self.l1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1) 

        # Q2 architecture with LayerNorm
        self.l4 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        s_emb = self.encoder(state)
        
        sa = torch.cat([s_emb, action], 1)
        
        q1 = F.relu(self.ln1(self.l1(sa)))
        q1 = F.relu(self.ln2(self.l2(q1)))
        q1 = self.l3(q1)

        q2 = F.relu(self.ln4(self.l4(sa)))
        q2 = F.relu(self.ln5(self.l5(q2)))
        q2 = self.l6(q2)
        return q1, q2

class Actor(nn.Module):
    """
    Gaussian Policy Network for SAC.
    
    Paper IV-B: Uses tanh squashing with proper Jacobian log-probability
    correction to handle high-dimensional distribution shift.
    
    Paper IV-A: Reparameterization trick for low-variance gradient estimates.
    """
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    
    def __init__(self, state_dim, action_dim, num_items=10, hidden_dim=256, micro_dim=6):
        super(Actor, self).__init__()
        self.encoder = StateEncoder(action_dim=0, num_items=num_items, hidden_dim=hidden_dim, micro_dim=micro_dim)
        
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        s_emb = self.encoder(state)
        
        x = F.relu(self.ln1(self.l1(s_emb)))
        x = F.relu(self.ln2(self.l2(x)))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        
        return mean, log_std

    def sample(self, state):
        """
        Paper IV-B: Reparameterization trick with tanh squashing.
        
        The tanh transformation induces a non-uniform distribution shift.
        We correct for this using the log-determinant of the Jacobian:
            log_prob -= log(1 - tanh(x)^2 + eps)
        
        This is critical for high-dimensional action spaces to avoid
        bias toward boundary actions.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick: x = mean + std * N(0,1)
        x_t = normal.rsample()
        
        # Tanh squashing to bound actions to [-1, 1]
        y_t = torch.tanh(x_t)
        action = y_t
        
        # Log probability with tanh squashing correction (Jacobian)
        # Paper IV-B: Corrects for non-uniform distribution shift
        log_prob = normal.log_prob(x_t)
        # Numerically stable tanh correction
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean
