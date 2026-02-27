"""
DOM-RL Training Script

Paper IV-E: Hybrid SAC-NSGA-II training loop.
Integrates periodic NSGA-II evolution for Pareto-optimal weight discovery
alongside standard SAC gradient-based policy optimization.
"""

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
    
    state_dim = 0 
    action_dim = env.action_space.shape[0] 
    num_items = env.num_categories 
    num_objectives = env.num_objectives
    
    print(f"Action Dimension: {action_dim}, Num Items: {num_items}, Num Objectives: {num_objectives}")

    # Initialize Agents
    agent = SACAgent(
        state_dim, action_dim, num_items=num_items, hidden_dim=512, 
        cql_weight=args.cql_weight, bc_weight=args.bc_weight
    )
    weight_agent = WeightAgent(
        action_dim=action_dim, num_items=num_items, hidden_dim=128,
        num_objectives=num_objectives,
        nsga2_pop_size=args.nsga2_pop_size,
        nsga2_generations=args.nsga2_generations
    )
    
    # Initialize Buffer (expanded dimensions)
    replay_buffer = ReplayBuffer(
        state_dim, action_dim, 
        micro_dim=6, num_objectives=num_objectives
    ) 
    
    total_steps = 0
    training_data = []
    
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
                pass
                
            print("Offline data loading skipped for V3 (Continuous Action mismatch). TODO: Implement Embedding Lookup.")
            
            offline_steps = 0

            for i in range(offline_steps):
                critic_loss, actor_loss, alpha, q_vals, entropy = agent.update(replay_buffer, args.batch_size)
                if i % 1000 == 0:
                    print(f"Offline Step {i}: Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}")
            
            total_steps = offline_steps
            print(f"Offline training done. Initializing total_steps to {total_steps}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Proceeding without offline training.")
    else:
        print("Dataset path not provided or not found. Skipping offline training.")
            
    # --- Online Training Loop ---
    start_steps = args.start_steps if total_steps == 0 else 0
    hit_rates = []
    
    for episode in range(args.max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_hits = 0
        episode_q_vals = []
        episode_actor_loss = []
        episode_critic_loss = []
        episode_entropy = []
        episode_reward_vectors = []
        
        step_count = 0
        for step in range(args.max_steps):
            step_count += 1
            
            # 1. Weight Agent Decision (Paper IV-E)
            weights = weight_agent.select_weights(state)
            env.weights = weights 
            state['weights'] = weights 
            
            # 2. SAC Agent Decision
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            
            # 3. Execute
            next_state, scalar_reward, terminated, truncated, info = env.step(action)
            reward_vec = info['reward_vector']
            
            done = terminated or truncated
            
            # Track reward vectors for NSGA-II
            weight_agent.record_reward_vector(reward_vec)
            episode_reward_vectors.append(reward_vec)
            
            # Track Accuracy
            try:
                if reward_vec[0] > 0.1: 
                    episode_hits += 1
            except IndexError:
                print(f"CRASH DEBUG: reward_vec type: {type(reward_vec)}")
                raise
            
            # 4. Scalarize Reward
            meta_reward = reward_vec[1] # Satisfaction as Meta-Reward
            
            # 5. Store
            replay_buffer.add(state, action, next_state, scalar_reward, done, meta_reward)
            
            state = next_state
            episode_reward += scalar_reward
            total_steps += 1
            
            # 6. Update Agents
            if total_steps >= args.update_after:
                critic_loss, actor_loss, alpha, q_vals, entropy = agent.update(replay_buffer, args.batch_size)
                episode_q_vals.extend(q_vals)
                episode_actor_loss.append(actor_loss)
                episode_critic_loss.append(critic_loss)
                episode_entropy.append(entropy)
                
                # Update Weight Agent
                w_critic_loss, w_actor_loss = weight_agent.update(replay_buffer, args.batch_size)
                
                if total_steps % 100 == 0:
                    writer.add_scalar("Loss/Actor", actor_loss, total_steps)
                    writer.add_scalar("Loss/Critic", critic_loss, total_steps)
                    writer.add_scalar("Loss/WeightCritic", w_critic_loss, total_steps)
                    writer.add_scalar("SAC/Alpha", alpha, total_steps)
                    writer.add_scalar("SAC/Entropy", entropy, total_steps)
            
            if done:
                break
        
        # --- Paper IV-E: Periodic NSGA-II Evolution ---
        if (episode + 1) % args.nsga2_evolve_interval == 0 and episode > 0:
            nsga2_weights = weight_agent.evolve_nsga2()
            hv = weight_agent.get_nsga2_hypervolume()
            writer.add_scalar("NSGA2/Hypervolume", hv, episode)
            if nsga2_weights is not None:
                print(f"  NSGA-II Evolution: Hypervolume={hv:.4f}, Best Weights={nsga2_weights[:3]}...")
        
        # Aggregated Episode Metrics
        avg_q = np.mean(episode_q_vals) if episode_q_vals else 0
        avg_actor_loss = np.mean(episode_actor_loss) if episode_actor_loss else 0
        avg_critic_loss = np.mean(episode_critic_loss) if episode_critic_loss else 0
        avg_entropy = np.mean(episode_entropy) if episode_entropy else 0
        final_satisfaction = env.user_satisfaction
        churn_prob = info.get('churn_probability', 0.0)
        
        # Calculate Hit Rate (Accuracy)
        ep_hit_rate = episode_hits / step_count
        hit_rates.append(ep_hit_rate)
        avg_hit_rate = np.mean(hit_rates[-100:])
        
        # Average reward vector components
        avg_reward_vec = np.mean(episode_reward_vectors, axis=0) if episode_reward_vectors else np.zeros(num_objectives)
        
        # Console Log with ACCURACY and churn
        print(f"Episode: {episode+1}/{args.max_episodes}, Reward: {episode_reward:.2f}, "
              f"Sat: {final_satisfaction:.2f}, Accuracy: {ep_hit_rate*100:.1f}%, "
              f"Churn: {churn_prob:.2f}")
        
        # TensorBoard Log
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Performance/Accuracy", ep_hit_rate, episode)
        writer.add_scalar("Env/Satisfaction", final_satisfaction, episode)
        writer.add_scalar("Env/ChurnProbability", churn_prob, episode)
        writer.add_scalar("SAC/AvgEntropy", avg_entropy, episode)
        
        # Per-objective logging
        obj_names = ["Engagement", "Satisfaction", "Diversity", "Fairness", "ChurnMitigation"]
        for i, name in enumerate(obj_names):
            if i < len(avg_reward_vec):
                writer.add_scalar(f"Objectives/{name}", avg_reward_vec[i], episode)
        
        # CSV Log Data
        training_data.append({
            "episode": episode + 1,
            "total_steps": total_steps,
            "reward": episode_reward,
            "accuracy": ep_hit_rate,
            "satisfaction": final_satisfaction,
            "churn_probability": churn_prob,
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "entropy": avg_entropy,
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
    from domrl.config import cfg
    parser = argparse.ArgumentParser(description="DOM-RL Training")
    parser.add_argument("--dataset_path", type=str, default=cfg.MOVIE_LENS_PATH, help="Path to MovieLens dataset")
    parser.add_argument("--max_episodes", type=int, default=2000, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--start_steps", type=int, default=5000, help="Random exploration steps")
    parser.add_argument("--update_after", type=int, default=1000, help="Steps before updates")
    parser.add_argument("--cql_weight", type=float, default=0.0, help="CQL Loss weight")
    parser.add_argument("--bc_weight", type=float, default=0.0, help="Behavior Cloning weight")
    parser.add_argument("--slate_size", type=int, default=3, help="Recommendation slate size")
    parser.add_argument("--offline_steps", type=int, default=5000, help="Offline gradient updates")
    # Paper IV-D: NSGA-II arguments
    parser.add_argument("--nsga2_pop_size", type=int, default=cfg.NSGA2_POP_SIZE, help="NSGA-II population size")
    parser.add_argument("--nsga2_generations", type=int, default=cfg.NSGA2_GENERATIONS, help="NSGA-II generations per evolution")
    parser.add_argument("--nsga2_evolve_interval", type=int, default=cfg.NSGA2_EVOLVE_INTERVAL, help="Episodes between NSGA-II evolutions")
    
    args = parser.parse_args()
    print(f"Training Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    train(args)
