import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100000):
        # state_dim is ignored or used as reference, but we use fixed keys for DOM-RL
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Storage for Dict State
        self.history = np.zeros((max_size, 10), dtype=np.int32)
        self.user_features = np.zeros((max_size, 2), dtype=np.float32)
        self.micro_signals = np.zeros((max_size, 3), dtype=np.float32)
        self.weights = np.zeros((max_size, 3), dtype=np.float32)
        
        # Storage for Next State
        self.next_history = np.zeros((max_size, 10), dtype=np.int32)
        self.next_user_features = np.zeros((max_size, 2), dtype=np.float32)
        self.next_micro_signals = np.zeros((max_size, 3), dtype=np.float32)
        self.next_weights = np.zeros((max_size, 3), dtype=np.float32)

        self.action = np.zeros((max_size, 1), dtype=np.int32) # Discrete action index
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        # Unpack State
        self.history[self.ptr] = state['history']
        self.user_features[self.ptr] = state['user_features']
        self.micro_signals[self.ptr] = state['micro_signals']
        self.weights[self.ptr] = state['weights']
        
        # Unpack Next State
        self.next_history[self.ptr] = next_state['history']
        self.next_user_features[self.ptr] = next_state['user_features']
        self.next_micro_signals[self.ptr] = next_state['micro_signals']
        self.next_weights[self.ptr] = next_state['weights']
        
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        # Helper to convert dict to tensors
        def make_state_dict(idx, is_next=False):
            if is_next:
                return {
                    "history": torch.LongTensor(self.next_history[idx]).to(self.device),
                    "user_features": torch.FloatTensor(self.next_user_features[idx]).to(self.device),
                    "micro_signals": torch.FloatTensor(self.next_micro_signals[idx]).to(self.device),
                    "weights": torch.FloatTensor(self.next_weights[idx]).to(self.device)
                }
            else:
                 return {
                    "history": torch.LongTensor(self.history[idx]).to(self.device),
                    "user_features": torch.FloatTensor(self.user_features[idx]).to(self.device),
                    "micro_signals": torch.FloatTensor(self.micro_signals[idx]).to(self.device),
                    "weights": torch.FloatTensor(self.weights[idx]).to(self.device)
                }

        return (
            make_state_dict(ind, is_next=False),
            torch.LongTensor(self.action[ind]).to(self.device),
            make_state_dict(ind, is_next=True),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
