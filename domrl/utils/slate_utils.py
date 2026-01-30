import itertools
import numpy as np

class SlateMapper:
    """
    Maps combinatorial slate tuples to discrete action indices.
    """
    def __init__(self, num_items=10, slate_size=3, allow_repeats=True):
        self.num_items = num_items
        self.slate_size = slate_size
        self.allow_repeats = allow_repeats
        
        # specific logic for 10 items, slate size 3
        # If order doesn't matter (set), use combinations
        # RecSys usually: Order matters (Ranking) or Set (Slate).
        # To reduce space size, let's assume Set (Order doesn't matter for now)
        # Combinations with replacement (if we can recommend 3 comedies)
        
        if allow_repeats:
            self.slates = list(itertools.combinations_with_replacement(range(num_items), slate_size))
        else:
            # combinations no replacement
            self.slates = list(itertools.combinations(range(num_items), slate_size))
            
        # Create Reverse Map
        self.slate_to_idx = {slate: i for i, slate in enumerate(self.slates)}
        
        self.num_actions = len(self.slates)
        print(f"SlateMapper Init: {num_items} items, size {slate_size}. Total Actions: {self.num_actions}")
        
    def get_slate(self, action_idx):
        return self.slates[action_idx]
    
    def get_action_idx(self, slate_tuple):
        # Sort tuple if we treat as set
        slate_tuple = tuple(sorted(slate_tuple))
        return self.slate_to_idx.get(slate_tuple, 0)
