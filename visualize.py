import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_latest_log_and_save():
    # Find latest csv
    list_of_files = glob.glob('logs/training_log_*.csv') 
    if not list_of_files:
        print("No log files found.")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Plotting data from {latest_file}")
    
    data = pd.read_csv(latest_file)
    
    # Create Figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'DOM-RL Training Metrics ({latest_file})', fontsize=16)
    
    # 1. Reward over Episodes
    axs[0, 0].plot(data['episode'], data['reward'], label='Episode Reward', color='blue', alpha=0.6)
    axs[0, 0].set_title('Reward per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')
    # Rolling average
    if len(data) > 10:
        axs[0, 0].plot(data['episode'], data['reward'].rolling(window=10).mean(), label='10-Ep Moving Avg', color='red')
    axs[0, 0].legend()
    
    # 2. Satisfaction
    axs[0, 1].plot(data['episode'], data['satisfaction'], label='Final Satisfaction', color='green', alpha=0.6)
    axs[0, 1].set_title('User Satisfaction (End of Episode)')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Satisfaction (0-1)')
    axs[0, 1].set_ylim(0, 1.1)
    
    # 3. Q-Value (Critic)
    if 'avg_q' in data.columns:
        axs[1, 0].plot(data['episode'], data['avg_q'], label='Avg Q-Value', color='purple')
        axs[1, 0].set_title('Average Q-Value (Critic Estimate)')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Q-Value')
    else:
        axs[1, 0].set_title("Avg Q-Value (Not Available)")
        axs[1, 0].text(0.5, 0.5, "Data not logged", ha='center')
    
    # 4. Losses
    axs[1, 1].plot(data['episode'], data['critic_loss'], label='Critic Loss', color='orange', alpha=0.7)
    axs[1, 1].plot(data['episode'], data['actor_loss'], label='Actor Loss', color='cyan', alpha=0.7)
    axs[1, 1].set_title('Network Losses')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('logs/training_summary.png')
    print("Saved plot to logs/training_summary.png")
    # plt.show() # Uncomment if running locally with display

if __name__ == "__main__":
    plot_latest_log_and_save()
