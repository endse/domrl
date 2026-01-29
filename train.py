import gymnasium as gym
import numpy as np
import torch
from domrl.env.rec_env import RealTimeRecEnv
from domrl.agent.sac import SACAgent
from domrl.agent.weight_agent import WeightAgent
from domrl.utils.replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
import time
from domrl.utils.data_loader import load_movielens_data

def train():
    # Create Environment
    env = RealTimeRecEnv()
    
    # State dim is strictly for compat, internal networks use dict
    state_dim = 0 
    action_dim = env.action_space.n
    
    # Initialize Agents
    agent = SACAgent(state_dim, action_dim)
    weight_agent = WeightAgent(action_dim=action_dim)
    
    # Initialize Buffer
    replay_buffer = ReplayBuffer(state_dim, 1) # Action dim 1 for discrete indices
    
    # Logging Setup
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"{log_dir}/tb_{timestamp}")
    csv_log_path = f"{log_dir}/training_log_{timestamp}.csv"
    
    # Training Config
    max_episodes = 50 # Reduced further for rapid verification
    max_steps = 100
    batch_size = 64
    start_steps = 5000 # Random exploration (50 episodes)
    update_after = 1000 # Start updating earlier
    
    total_steps = 0
    training_data = [] # List to store dicts for CSV
    
    print(f"Starting Training... Logging to {log_dir}")

    # --- Offline Data Loading ---
    dataset_path = "c:\\Users\\cy569\\Downloads\\ml-latest\\dataset"
    if os.path.exists(dataset_path):
        transitions = load_movielens_data(dataset_path, history_len=10, max_rows=100000)
        print(f"Populating ReplayBuffer with {len(transitions)} transitions...")
        for t in transitions:
            state, action, next_state, reward, done = t
            replay_buffer.add(state, action, next_state, reward, done)
        
        # Offline Training Phase
        print("Starting Offline Training Phase...")
        offline_steps = 5000
        for i in range(offline_steps):
            critic_loss, actor_loss, alpha, q_vals = agent.update(replay_buffer, batch_size)
            if i % 1000 == 0:
                print(f"Offline Step {i}: Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}")
    else:
        print("Dataset not found. Skipping offline training.")
    
    # Reset total_steps for online phase logging consistency (or keep it cumulative)
    if os.path.exists(dataset_path):
        total_steps = offline_steps
        print(f"Skipping random start steps. Initializing total_steps to {total_steps}")
    else:
        total_steps = 0 
    
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_q_vals = []
        episode_actor_loss = []
        episode_critic_loss = []
        
        for step in range(max_steps):
            
            # 1. Weight Agent Decision
            weights = weight_agent.select_weights(state)
            env.weights = weights # Update Environment
            state['weights'] = weights # Update Observation
            
            # 2. SAC Agent Decision
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            
            # 3. Execute
            next_state, reward_vec, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 4. Scalarize Reward
            scalar_reward = np.dot(reward_vec, weights)
            
            # 5. Store
            replay_buffer.add(state, action, next_state, scalar_reward, done)
            
            state = next_state
            episode_reward += scalar_reward
            total_steps += 1
            
            # 6. Update Agents
            if total_steps >= update_after and total_steps % 1 == 0:
                critic_loss, actor_loss, alpha, q_vals = agent.update(replay_buffer, batch_size)
                episode_q_vals.extend(q_vals)
                episode_actor_loss.append(actor_loss)
                episode_critic_loss.append(critic_loss)
                
                # Update Weight Agent (Placeholder / Meta-Reward logic would go here)
                # weight_agent.update(...)
                
                # Scalar logs per step (optional, can be noisy)
                if total_steps % 100 == 0:
                    writer.add_scalar("Loss/Actor", actor_loss, total_steps)
                    writer.add_scalar("Loss/Critic", critic_loss, total_steps)
            
            if done:
                break
        
        # Aggregated Episode Metrics
        avg_q = np.mean(episode_q_vals) if episode_q_vals else 0
        avg_actor_loss = np.mean(episode_actor_loss) if episode_actor_loss else 0
        avg_critic_loss = np.mean(episode_critic_loss) if episode_critic_loss else 0
        final_satisfaction = env.user_satisfaction
        
        # Console Log
        # Weights are from the last step, but gives an idea
        print(f"Episode: {episode+1}, Reward: {episode_reward:.2f}, AvgQ: {avg_q:.2f}, Sat: {final_satisfaction:.2f}, W: {weights.round(2)}")
        
        # TensorBoard Log
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Value/AvgQ", avg_q, episode)
        writer.add_scalar("Env/Satisfaction", final_satisfaction, episode)
        
        # CSV Log Data
        training_data.append({
            "episode": episode + 1,
            "total_steps": total_steps,
            "reward": episode_reward,
            "avg_q": avg_q,
            "satisfaction": final_satisfaction,
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "w_eng": weights[0],
            "w_sat": weights[1],
            "w_div": weights[2]
        })
        
        if (episode+1) % 50 == 0:
             torch.save(agent.actor.state_dict(), f"{log_dir}/actor_{episode+1}.pth")
             # Save CSV checkpoint
             pd.DataFrame(training_data).to_csv(csv_log_path, index=False)

    # Final Save
    torch.save(agent.actor.state_dict(), f"{log_dir}/actor_final.pth")
    pd.DataFrame(training_data).to_csv(csv_log_path, index=False)
    writer.close()
    print("Training Complete.")

if __name__ == "__main__":
    train()
