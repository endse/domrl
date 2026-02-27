"""
Replay Buffer for DOM-RL

Stores transitions with dictionary-based states for the SAC and Weight agents.
Paper III-A: Now supports 6-dimensional micro-signals and 5-dimensional weights.
"""

import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100000, 
                 micro_dim=6, num_objectives=5):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Storage for Dict State
        self.history = np.zeros((max_size, 10), dtype=np.int32)
        self.user_features = np.zeros((max_size, 2), dtype=np.float32)
        self.micro_signals = np.zeros((max_size, micro_dim), dtype=np.float32)  # 3 -> 6
        self.weights = np.zeros((max_size, num_objectives), dtype=np.float32)   # 4 -> 5
        self.persona_id = np.zeros((max_size, 1), dtype=np.int32)
        
        # Storage for Next State
        self.next_history = np.zeros((max_size, 10), dtype=np.int32)
        self.next_user_features = np.zeros((max_size, 2), dtype=np.float32)
        self.next_micro_signals = np.zeros((max_size, micro_dim), dtype=np.float32)
        self.next_weights = np.zeros((max_size, num_objectives), dtype=np.float32)
        self.next_persona_id = np.zeros((max_size, 1), dtype=np.int32)

        self.action_dim = action_dim
        
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.meta_reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done, meta_reward=0.0):
        # Unpack State
        self.history[self.ptr] = state['history']
        self.user_features[self.ptr] = state['user_features']
        self.micro_signals[self.ptr] = state['micro_signals']
        self.weights[self.ptr] = state['weights']
        self.persona_id[self.ptr] = state.get('persona_id', 0)
        
        # Unpack Next State
        self.next_history[self.ptr] = next_state['history']
        self.next_user_features[self.ptr] = next_state['user_features']
        self.next_micro_signals[self.ptr] = next_state['micro_signals']
        self.next_weights[self.ptr] = next_state['weights']
        self.next_persona_id[self.ptr] = next_state.get('persona_id', 0)
        
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.meta_reward[self.ptr] = meta_reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        def make_state_dict(idx, is_next=False):
            if is_next:
                return {
                    "history": torch.LongTensor(self.next_history[idx]).to(self.device),
                    "user_features": torch.FloatTensor(self.next_user_features[idx]).to(self.device),
                    "micro_signals": torch.FloatTensor(self.next_micro_signals[idx]).to(self.device),
                    "weights": torch.FloatTensor(self.next_weights[idx]).to(self.device),
                    "persona_id": torch.LongTensor(self.next_persona_id[idx]).to(self.device).squeeze(-1)
                }
            else:
                 return {
                    "history": torch.LongTensor(self.history[idx]).to(self.device),
                    "user_features": torch.FloatTensor(self.user_features[idx]).to(self.device),
                    "micro_signals": torch.FloatTensor(self.micro_signals[idx]).to(self.device),
                    "weights": torch.FloatTensor(self.weights[idx]).to(self.device),
                    "persona_id": torch.LongTensor(self.persona_id[idx]).to(self.device).squeeze(-1)
                }

        return (
            make_state_dict(ind, is_next=False),
            torch.FloatTensor(self.action[ind]).to(self.device),
            make_state_dict(ind, is_next=True),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.meta_reward[ind]).to(self.device)
        )
