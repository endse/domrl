import gymnasium as gym
import numpy as np
import torch
from domrl.env.rec_env import RealTimeRecEnv
from domrl.agent.sac import SACAgent
import pandas as pd
import os
import glob

def load_latest_actor(agent, log_dir="logs"):
    files = glob.glob(f'{log_dir}/actor_*.pth')
    if not files:
        # Fallback to final if exists
        if os.path.exists(f'{log_dir}/actor_final.pth'):
             agent.actor.load_state_dict(torch.load(f'{log_dir}/actor_final.pth'))
             print(f"Loading checkpoint: actor_final.pth")
             return
        print("No checkpoints found. Running with random weights.")
        return
    latest = max(files, key=os.path.getctime)
    print(f"Loading checkpoint: {latest}")
    try:
        agent.actor.load_state_dict(torch.load(latest))
    except Exception as e:
        print(f"Failed to load checkpoint {latest}: {e}")
        print("Falling back to random weights.")

def collect_data(episodes=100):
    env = RealTimeRecEnv()
    state_dim = 0 # Dict compatibility
    action_dim = env.action_space.n
    agent = SACAgent(state_dim, action_dim, hidden_dim=512)
    
    load_latest_actor(agent)
    
    data = []
    
    print(f"Collecting data from {episodes} episodes on {agent.device}...")
    
    for ep in range(episodes):
        state, _ = env.reset()
        step = 0
        
        # Scenario Labeling
        # user_features: [Enthusiasm, Time]
        # micro_signals: [Scroll, Hover, View]
        # weights: [w_eng, w_sat, w_div, w_fair]
        
        weights = state['weights']
        w_eng, w_sat, w_div, w_fair = weights
        scenario = "Balanced"
        if w_div > 2.0: scenario = "Safety" 
        if w_eng > 1.2 and w_div < 1.0: scenario = "Growth"
        if w_fair > 2.0: scenario = "Fairness"
        
        while True:
            # For visualization, we might want deterministic or sampled actions
            # Let's use evaluate=False to get some variation distribution
            action = agent.select_action(state, evaluate=False)
            
            # Log Data BEFORE step (State S, Action A)
            obs_entry = {
                "Episode": ep,
                "Step": step,
                "Enthusiasm": state['user_features'][0],
                "TimeOfDay": state['user_features'][1],
                "ScrollVel": state['micro_signals'][0],
                "Hover": state['micro_signals'][1],
                "ViewTime": state['micro_signals'][2],
                "w_Eng": w_eng,
                "w_Sat": w_sat,
                "w_Div": w_div,
                "w_Fair": w_fair,
                "Scenario": scenario,
                "Action": action
            }
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Log Data AFTER step (Reward R, Sat S')
            # Reward is a vector [r_eng, r_sat, r_div, r_fair]
            scalar_reward = np.dot(reward, [w_eng, w_sat, w_div, w_fair])
            obs_entry["Reward"] = scalar_reward
            obs_entry["Satisfaction"] = env.user_satisfaction
            obs_entry["Done"] = terminated or truncated
            
            data.append(obs_entry)
            
            state = next_state
            step += 1
            
            if terminated or truncated:
                break
                
    df = pd.DataFrame(data)
    df.to_csv("logs/granular_history.csv", index=False)
    print(f"Data collection complete. Saved {len(df)} rows to logs/granular_history.csv")

if __name__ == "__main__":
    collect_data()
