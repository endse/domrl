import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from domrl.env.rec_env import RealTimeRecEnv
from domrl.agent.sac import SACAgent
from domrl.agent.weight_agent import WeightAgent
from domrl.utils.replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
import time
import argparse
from domrl.utils.data_loader import load_movielens_data

def train(args):
    # Setup Paths
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"{log_dir}/tb_{timestamp}")
    csv_log_path = f"{log_dir}/training_log_{timestamp}.csv"
    
    print(f"Starting Training... Logging to {log_dir}")
    print(f"Configuration: {args}")

    # Create Environment
    env = RealTimeRecEnv(slate_size=args.slate_size)
    
    # State dim is strictly for compat, internal networks use dict
    state_dim = 0 
    action_dim = env.action_space.n
    
    # Initialize Agents (Bigger Capacity for High Accuracy)
    agent = SACAgent(state_dim, action_dim, hidden_dim=512, cql_weight=args.cql_weight, bc_weight=args.bc_weight)
    weight_agent = WeightAgent(action_dim=action_dim, hidden_dim=128)
    
    # Initialize Buffer
    replay_buffer = ReplayBuffer(state_dim, 1) # Action dim 1 for discrete indices
    
    total_steps = 0
    training_data = [] # List to store dicts for CSV
    
    # --- Offline Data Loading ---
    dataset_path = args.dataset_path
    if dataset_path and os.path.exists(dataset_path):
        try:
            full_path = os.path.abspath(dataset_path)
            print(f"Loading dataset from: {full_path}")
            transitions = load_movielens_data(full_path, history_len=10, max_rows=100000)
            print(f"Populating ReplayBuffer with {len(transitions)} transitions...")
            for t in transitions:
                state, action, next_state, reward, done = t
                replay_buffer.add(state, action, next_state, reward, done)
            
            # Offline Training Phase
            print("Starting Offline Training Phase...")
            offline_steps = 5000
            for i in range(offline_steps):
                critic_loss, actor_loss, alpha, q_vals = agent.update(replay_buffer, args.batch_size)
                if i % 1000 == 0:
                    print(f"Offline Step {i}: Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}")
            
            # Skip random exploration if we have offline data
            total_steps = offline_steps
            print(f"Offline training done. Initializing total_steps to {total_steps}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Proceeding without offline training.")
    else:
        print("Dataset path not provided or not found. Skipping offline training.")
            
    # --- Online Training Loop ---
    start_steps = args.start_steps if total_steps == 0 else 0
    hit_rates = [] # Track accuracy
    
    for episode in range(args.max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_hits = 0 # Count successful recommendations
        episode_q_vals = []
        episode_actor_loss = []
        episode_critic_loss = []
        
        step_count = 0
        for step in range(args.max_steps):
            step_count += 1
            
            # ... (Agents Decision) ...
            # 1. Weight Agent Decision
            weights = weight_agent.select_weights(state)
            env.weights = weights 
            state['weights'] = weights 
            
            # 2. SAC Agent Decision
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            
            # 3. Execute
            next_state, reward_vec, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Track Accuracy (Hit: Did the user click? i.e., Engagement Reward > 0)
            if reward_vec[0] > 0.1: 
                episode_hits += 1
            
            # ... (Rest of loop) ...
            # 4. Scalarize Reward
            scalar_reward = np.dot(reward_vec, weights)
            meta_reward = reward_vec[1] # Satisfaction as Meta-Reward
            
            # 5. Store
            replay_buffer.add(state, action, next_state, scalar_reward, done, meta_reward)
            
            state = next_state
            episode_reward += scalar_reward
            total_steps += 1
            
            # 6. Update Agents
            if total_steps >= args.update_after:
                critic_loss, actor_loss, alpha, q_vals = agent.update(replay_buffer, args.batch_size)
                episode_q_vals.extend(q_vals)
                episode_actor_loss.append(actor_loss)
                episode_critic_loss.append(critic_loss)
                
                # Update Weight Agent
                w_critic_loss, w_actor_loss = weight_agent.update(replay_buffer, args.batch_size)
                
                if total_steps % 100 == 0:
                    writer.add_scalar("Loss/Actor", actor_loss, total_steps)
                    writer.add_scalar("Loss/Critic", critic_loss, total_steps)
                    writer.add_scalar("Loss/WeightCritic", w_critic_loss, total_steps)
            
            if done:
                break
        
        # Aggregated Episode Metrics
        avg_q = np.mean(episode_q_vals) if episode_q_vals else 0
        avg_actor_loss = np.mean(episode_actor_loss) if episode_actor_loss else 0
        avg_critic_loss = np.mean(episode_critic_loss) if episode_critic_loss else 0
        final_satisfaction = env.user_satisfaction
        
        # Calculate Hit Rate (Accuracy)
        ep_hit_rate = episode_hits / step_count
        hit_rates.append(ep_hit_rate)
        avg_hit_rate = np.mean(hit_rates[-100:]) # Rolling average
        
        # Console Log with ACCURACY
        print(f"Episode: {episode+1}/{args.max_episodes}, Reward: {episode_reward:.2f}, Sat: {final_satisfaction:.2f}, Accuracy (HitRate): {ep_hit_rate*100:.1f}%")
        
        # TensorBoard Log
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Performance/Accuracy", ep_hit_rate, episode)
        writer.add_scalar("Env/Satisfaction", final_satisfaction, episode)
        
        # CSV Log Data
        training_data.append({
            "episode": episode + 1,
            "total_steps": total_steps,
            "reward": episode_reward,
            "accuracy": ep_hit_rate,
            "satisfaction": final_satisfaction,
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss
        })
        
        if (episode+1) % 50 == 0:
             torch.save(agent.actor.state_dict(), f"{log_dir}/actor_{episode+1}.pth")
             pd.DataFrame(training_data).to_csv(csv_log_path, index=False)

    # Final Save
    torch.save(agent.actor.state_dict(), f"{log_dir}/actor_final.pth")
    pd.DataFrame(training_data).to_csv(csv_log_path, index=False)
    writer.close()
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DOM-RL Training")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to MovieLens dataset for offline pre-training")
    parser.add_argument("--max_episodes", type=int, default=50, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for updates")
    parser.add_argument("--start_steps", type=int, default=5000, help="Steps for random exploration")
    parser.add_argument("--update_after", type=int, default=1000, help="Steps before starting updates")
    parser.add_argument("--cql_weight", type=float, default=0.0, help="Weight for CQL Loss")
    parser.add_argument("--bc_weight", type=float, default=0.0, help="Weight for Behavior Cloning Loss")
    parser.add_argument("--slate_size", type=int, default=3, help="Size of recommendation slate")
    
    args = parser.parse_args()
    train(args)
