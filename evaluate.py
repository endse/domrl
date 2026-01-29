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
        print("No checkpoints found. Running with random weights (Expect poor performance).")
        return
    latest = max(files, key=os.path.getctime)
    print(f"Loading checkpoint: {latest}")
    agent.actor.load_state_dict(torch.load(latest))

def run_scenario(env, agent, weights, name, episodes=50):
    print(f"\n--- Running Scenario: {name} ---")
    print(f"Weights: Eng={weights[0]}, Sat={weights[1]}, Churn={weights[2]}")
    
    results = []
    
    for ep in range(episodes):
        state, _ = env.reset(options={'weights': weights})
        episode_reward = 0
        steps = 0
        
        while True:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Scalarize reward for evaluation metric
            episode_reward += np.dot(reward, weights)
            state = next_state
            steps += 1
            
            if terminated or truncated:
                break
                
        # Metric: Did we churn? (If steps < max_steps, likely churned)
        churned = 1 if steps < env.max_steps else 0
        term_reason = "Churn" if churned else "MaxSteps"
        
        results.append({
            "scenario": name,
            "reward": episode_reward,
            "satisfaction": env.user_satisfaction,
            "final_satisfaction": env.user_satisfaction,
            "churned": churned, 
            "term_reason": term_reason,
            "steps": steps
        })
        
    df = pd.DataFrame(results)
    print(f"Avg Reward: {df['reward'].mean():.2f}")
    print(f"Avg Satisfaction: {df['satisfaction'].mean():.2f}")
    print(f"Churn Rate: {df['churned'].mean():.2f}")
    return df

def evaluate():
    env = RealTimeRecEnv()
    state_dim = 0
    action_dim = env.action_space.n
    agent = SACAgent(state_dim, action_dim)
    
    load_latest_actor(agent)
    
    # Define Scenarios
    # 1. Growth: High Engagement Value, Low Churn Penalty
    #    Agent should be aggressive, maybe riskier.
    scenarios = [
        ("Growth (High Eng)", [2.0, 0.5, 0.5]),
        ("Safety (High Churn)", [0.5, 0.5, 5.0]),
        ("Balanced", [1.0, 0.5, 2.0])
    ]
    
    all_results = []
    for name, w in scenarios:
        df = run_scenario(env, agent, w, name)
        all_results.append(df)
        
    final_df = pd.concat(all_results)
    final_df.to_csv("logs/evaluation_results_hybrid.csv", index=False)
    print("\nSaved detailed results to logs/evaluation_results_hybrid.csv")

if __name__ == "__main__":
    evaluate()
