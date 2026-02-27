"""
DOM-RL v2/v3 Verification Script (Updated for Paper Alignment)

Compatible with the expanded DOM-RL architecture:
- 6-dimensional micro-behavioral signals
- 5-objective reward vector
- Hybrid SAC-NSGA-II weight agent
"""

import numpy as np
import torch
import os
from domrl.env.rec_env import RealTimeRecEnv
from domrl.agent.sac import SACAgent
from domrl.agent.weight_agent import WeightAgent
from domrl.utils.replay_buffer import ReplayBuffer

def verify_system():
    print("Initializing Environment...")
    env = RealTimeRecEnv()
    obs, _ = env.reset()
    
    print("Observation Keys:", obs.keys())
    print("Action Space:", env.action_space)
    print(f"Micro-signals shape: {obs['micro_signals'].shape}")
    print(f"Weights shape: {obs['weights'].shape}")
    
    action_dim = env.action_space.shape[0]
    state_dim = 0
    
    print(f"Action Dim: {action_dim}")
    
    print("Initializing Agents...")
    sac_agent = SACAgent(state_dim, action_dim, num_items=env.num_categories)
    weight_agent = WeightAgent(
        action_dim=action_dim, num_items=env.num_categories,
        num_objectives=env.num_objectives
    )
    
    replay_buffer = ReplayBuffer(
        state_dim, action_dim, max_size=1000,
        micro_dim=6, num_objectives=env.num_objectives
    )
    
    print("Running Simulation Loop...")
    for i in range(5):
        # Select Actions
        action_emb = sac_agent.select_action(obs)
        weights = weight_agent.select_weights(obs)
        
        # Apply WeightAgent's action to Env
        env.weights = weights
        obs['weights'] = weights
        
        # Step Env
        next_obs, reward, done, truncated, info = env.step(action_emb)
        
        # Store in Buffer
        replay_buffer.add(obs, action_emb, next_obs, reward, done, meta_reward=float(reward)) 
        
        obs = next_obs
        if done: obs, _ = env.reset()
        
    print(f"Buffer Size: {replay_buffer.size}")
        
    print("Running Agent Updates...")
    if replay_buffer.size >= 2: 
        critic_loss, actor_loss, alpha, q_vals, entropy = sac_agent.update(replay_buffer, batch_size=2)
        w_critic_loss, w_actor_loss = weight_agent.update(replay_buffer, batch_size=2)
        print(f"SAC: critic={critic_loss:.4f}, actor={actor_loss:.4f}, alpha={alpha:.4f}, entropy={entropy:.4f}")
        print(f"Weight: critic={w_critic_loss:.4f}, actor={w_actor_loss:.4f}")
        print("Updates Successful")
    else:
        print("Skipping Update (Buffer too small)")

    print("Verification Complete!")

if __name__ == "__main__":
    verify_system()
