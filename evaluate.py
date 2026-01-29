import gymnasium as gym
import numpy as np
import torch
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from domrl.env.rec_env import RealTimeRecEnv
from domrl.agent.sac import SACAgent
from domrl.agent.baselines import RandomAgent, StaticAgent

def load_latest_actor(agent, log_dir="logs"):
    files = glob.glob(f'{log_dir}/actor_*.pth')
    
    # Try final first
    if os.path.exists(f'{log_dir}/actor_final.pth'):
         print(f"Loading checkpoint: actor_final.pth")
         agent.actor.load_state_dict(torch.load(f'{log_dir}/actor_final.pth'))
         return

    if not files:
        print("No checkpoints found. Running with random weights (Expect poor performance).")
        return
    latest = max(files, key=os.path.getctime)
    print(f"Loading checkpoint: {latest}")
    agent.actor.load_state_dict(torch.load(latest))

def run_agent_eval(env, agent, agent_name, scenarios, episodes=20):
    results = []
    
    for sc_name, weights in scenarios:
        # print(f"Evaluating {agent_name} on {sc_name}...")
        
        for ep in range(episodes):
            state, _ = env.reset(options={'weights': weights})
            episode_reward = 0
            steps = 0
            
            while True:
                # Handle different agent signatures if needed, but here all match
                action = agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # Scalarize reward
                episode_reward += np.dot(reward, weights)
                state = next_state
                steps += 1
                
                if terminated or truncated:
                    break
            
            churned = 1 if steps < env.max_steps else 0
            
            results.append({
                "Agent": agent_name,
                "Scenario": sc_name,
                "Reward": episode_reward,
                "Satisfaction": env.user_satisfaction,
                "Steps": steps,
                "Churned": churned
            })
            
    return results

def evaluate():
    env = RealTimeRecEnv()
    action_dim = env.action_space.n
    
    # Initialize Agents
    sac_agent = SACAgent(0, action_dim)
    load_latest_actor(sac_agent)
    
    random_agent = RandomAgent(action_dim)
    static_agent = StaticAgent(action_dim, preferred_action=0) # Always recommend Action 0
    
    agents = {
        "SAC (Ours)": sac_agent,
        "Random": random_agent,
        "Static (Act 0)": static_agent
    }
    
    # Define Scenarios
    scenarios = [
        ("Balanced", [1.0, 0.5, 2.0]),
        ("Growth", [2.0, 0.2, 0.5]),
        ("Safety", [0.5, 0.8, 5.0])
    ]
    
    print(f"Starting Benchmark on {len(scenarios)} scenarios with {len(agents)} agents...")
    
    all_data = []
    for name, agent in agents.items():
        data = run_agent_eval(env, agent, name, scenarios)
        all_data.extend(data)
        
    df = pd.DataFrame(all_data)
    df.to_csv("logs/benchmark_results.csv", index=False)
    print("Benchmark complete. Results saved to logs/benchmark_results.csv")
    
    # --- Visualization ---
    print("Generating Benchmark Plot...")
    plt.figure(figsize=(12, 6))
    
    # 1. Rewards
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x='Scenario', y='Reward', hue='Agent', palette='magma')
    plt.title("Reward Comparison")
    
    # 2. Satisfaction
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x='Scenario', y='Satisfaction', hue='Agent', palette='magma')
    plt.title("User Satisfaction Comparison")
    
    plt.tight_layout()
    plt.savefig("logs/benchmark_summary.png")
    print("Plot saved to logs/benchmark_summary.png")
    
    # Console Summary
    summary = df.groupby(["Agent", "Scenario"])[["Reward", "Satisfaction", "Churned"]].mean()
    print("\nBenchmark Summary:")
    print(summary)

if __name__ == "__main__":
    evaluate()
