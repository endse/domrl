import numpy as np
import torch

class RandomAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def select_action(self, state, evaluate=True):
        return np.random.randint(0, self.action_dim)

class StaticAgent:
    def __init__(self, action_dim, preferred_action=0):
        self.action_dim = action_dim
        self.preferred_action = preferred_action

    def select_action(self, state, evaluate=True):
        # Always returns the same action
        return self.preferred_action
