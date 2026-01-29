import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class StateEncoder(nn.Module):
    def __init__(self, action_dim=10, hidden_dim=64):
        super(StateEncoder, self).__init__()
        
        # History Encoder
        # Action IDs 0-9. Embedding needs 10 + 1 (for potential padding/start?)
        self.embedding = nn.Embedding(num_embeddings=action_dim + 1, embedding_dim=16)
        self.gru = nn.GRU(input_size=16, hidden_size=32, batch_first=True)
        
        # Persona Encoder
        self.persona_embedding = nn.Embedding(num_embeddings=4, embedding_dim=4)
        
        # Combined Feature Dimension
        # GRU_Out(32) + User(2) + Micro(3) + Weights(3) + Persona(4) = 44
        self.fc = nn.Linear(32 + 2 + 3 + 3 + 4, hidden_dim)
        
    def forward(self, state_dict):
        # 1. History -> GRU
        h = state_dict['history'] # (B, 10)
        emb = self.embedding(h)   # (B, 10, 16)
        _, h_n = self.gru(emb)    # h_n: (1, B, 32)
        h_seq = h_n.squeeze(0)    # (B, 32)
        
        # 2. Other features
        u = state_dict['user_features']
        m = state_dict['micro_signals']
        w = state_dict['weights']
        p = state_dict['persona_id']
        
        if p.dim() == 2 and p.shape[1] == 1:
             p = p.squeeze(1) # Ensure (B,) for embedding
        
        p_emb = self.persona_embedding(p) # (B, 4)
        
        # 3. Concatenate
        cat = torch.cat([h_seq, u, m, w, p_emb], dim=1) # (B, 44)
        
        # 4. Dense Encode
        x = F.relu(self.fc(cat))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Note: state_dim ignored, using encoder
        self.encoder = StateEncoder(action_dim=action_dim, hidden_dim=hidden_dim)
        
        # Q1 architecture
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim) 

        # Q2 architecture
        self.l4 = nn.Linear(hidden_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        s_emb = self.encoder(state)
        
        q1 = F.relu(self.l1(s_emb))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_emb))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.encoder = StateEncoder(action_dim=action_dim, hidden_dim=hidden_dim)
        
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        s_emb = self.encoder(state)
        
        x = F.relu(self.l1(s_emb))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        probs = F.softmax(x, dim=-1)
        return probs

    def sample(self, state):
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        z = (probs == 0.0).float() * 1e-8
        log_probs = torch.log(probs + z)
        
        return action, probs, log_probs
