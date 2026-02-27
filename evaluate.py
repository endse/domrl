import gymnasium as gym
import numpy as np
import torch
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from domrl.env.rec_env import RealTimeRecEnv
from domrl.agent.sac import SACAgent
from domrl.agent.baselines import RandomAgent, StaticAgent
from domrl.utils.data_loader import load_movielens_data
from torch.distributions import Categorical

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

class DoublyRobustEvaluator:
    def __init__(self, dataset_path, action_dim=10):
        self.dataset_path = dataset_path
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Data
        self.transitions = load_movielens_data(dataset_path, max_rows=50000)
        
        # Estimate Behavior Policy (Marginal Counts)
        # pi_b(a|s) approx P(a) (Naïve) or Conditional via Histogram/NN
        # For simplicity, we use marginal probabilities (Action Popularity)
        print("Estimating Behavior Policy...")
        counts = np.zeros(action_dim)
        for t in self.transitions:
            _, action, _, _, _ = t
            counts[action] += 1
        probs = counts / counts.sum()
        self.behavior_probs = torch.FloatTensor(probs).to(self.device)
        print(f"Behavior Policy (Marginal): {probs}")
        
    def evaluate(self, agent):
        """
        Compute Doubly Robust Estimate:
        DR = V_model + rho * (r - Q(s,a_obs))
        """
        
        total_dr_value = 0
        count = 0
        
        for t in self.transitions:
            state, action, next_state, reward, done = t
            
            # 1. Get Agent Action Probs pi(a|s)
            state_tensor = {}
            for k, v in state.items():
                if k == 'history' or k == 'persona_id':
                     t_v = torch.as_tensor(v, device=self.device, dtype=torch.long)
                     if t_v.dim() == 0: t_v = t_v.unsqueeze(0)
                     if t_v.dim() == 1 and k=='history': t_v = t_v.unsqueeze(0)
                     state_tensor[k] = t_v
                else:
                    t_v = torch.as_tensor(v, device=self.device, dtype=torch.float32)
                    if t_v.dim() == 1: t_v = t_v.unsqueeze(0)
                    state_tensor[k] = t_v
            
            with torch.no_grad():
                 _, probs, _ = agent.actor.sample(state_tensor) # (1, A)
                 q1, q2 = agent.critic(state_tensor)
                 q = torch.min(q1, q2) # (1, A)
                 
            pi_a = probs[0, action].item()
            pi_b = self.behavior_probs[action].item() + 1e-6
            
            rho = pi_a / pi_b
            rho = min(rho, 10.0) # Clipping
            
            # Model Value V(s) = sum(pi(a'|s) * Q(s, a'))
            v_model = torch.sum(probs * q, dim=1).item()
            
            # Q(s, a_observed)
            q_obs = q[0, action].item()
            
            # DR = V_model + rho * (r - Q_obs)
            dr_est = v_model + rho * (reward - q_obs)
            
            total_dr_value += dr_est
            count += 1
            
        return total_dr_value / count

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

def evaluate(args):
    dataset_path = args.dataset_path
    
    print(f"Evaluation Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    env = RealTimeRecEnv(slate_size=args.slate_size)
    action_dim = env.action_space.n
    
    # Initialize Agents
    sac_agent = SACAgent(0, action_dim, num_items=env.num_categories, hidden_dim=512) # State dim 0 as placeholder
    load_latest_actor(sac_agent)
    
    random_agent = RandomAgent(action_dim)
    static_agent = StaticAgent(action_dim, preferred_action=0) # Always recommend Action 0
    
    agents = {
        "SAC (Ours)": sac_agent,
        "Random": random_agent,
        "Static (Act 0)": static_agent
    }
    
    # --- Offline Doubly Robust Evaluation ---
    if dataset_path and os.path.exists(dataset_path):
        print("\n--- Starting Doubly Robust Evaluation (Offline) ---")
        dr_evaluator = DoublyRobustEvaluator(dataset_path, action_dim)
        dr_results = []
        for name, agent in agents.items():
            if name == "SAC (Ours)":
                score = dr_evaluator.evaluate(agent)
                print(f"Agent: {name}, DR Score: {score:.4f}")
                dr_results.append({"Agent": name, "DR_Score": score})
        
        pd.DataFrame(dr_results).to_csv("logs/dr_results.csv", index=False)
        print("DR Results saved to logs/dr_results.csv")
    
    # --- Online Simulation Evaluation ---
    # Define Scenarios
    scenarios = [
        ("Balanced", [1.0, 0.5, 2.0, 1.0]),
        ("Growth", [2.0, 0.2, 0.5, 0.0]),
        ("Safety", [0.5, 0.8, 5.0, 0.0]),
        ("Fairness", [0.5, 0.5, 1.0, 5.0])
    ]
    
    print(f"\n--- Starting Online Benchmark on {len(scenarios)} scenarios ---")
    
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
    from domrl.config import cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=cfg.MOVIE_LENS_PATH, help="Path to MovieLens dataset for DR Eval")
    parser.add_argument("--slate_size", type=int, default=3, help="Size of recommendation slate")
    args = parser.parse_args()
    evaluate(args)
