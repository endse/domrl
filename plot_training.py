import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_latest_log():
    # Find latest log file
    list_of_files = glob.glob('logs/training_log_*.csv') 
    if not list_of_files:
        print("No log files found in logs/")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Plotting data from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    if df.empty:
        print("Log file is empty.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # 1. Rewards & Satisfaction
    ax1 = axes[0]
    ax1.plot(df['total_steps'], df['reward'], label='Scalar Reward', alpha=0.6)
    ax1.plot(df['total_steps'], df['satisfaction'] * 10, label='Satisfaction (x10)', linewidth=2)
    ax1.set_ylabel('Reward / Sat')
    ax1.set_title('Training Performance')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Weights
    ax2 = axes[1]
    ax2.plot(df['total_steps'], df['w_eng'], label='Weight: Engagement')
    ax2.plot(df['total_steps'], df['w_sat'], label='Weight: Satisfaction')
    ax2.plot(df['total_steps'], df['w_div'], label='Weight: Diversity')
    ax2.set_ylabel('Weight Value')
    ax2.set_title('Dynamic Weights')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Losses
    ax3 = axes[2]
    # Check if columns exist (might be missing in early logs)
    if 'Loss/WeightActor' in df.columns:
         ax3.plot(df['total_steps'], df['Loss/WeightActor'], label='Weight Actor Loss')
         ax3.plot(df['total_steps'], df['Loss/WeightCritic'], label='Meta Critic Loss')
    elif 'actor_loss' in df.columns:
         ax3.plot(df['total_steps'], df['actor_loss'], label='SAC Actor Loss')
         ax3.plot(df['total_steps'], df['critic_loss'], label='SAC Critic Loss')
         
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Losses')
    ax3.legend()
    ax3.grid(True)
    
    plt.xlabel('Total Steps')
    plt.tight_layout()
    
    output_path = latest_file.replace('.csv', '_plot.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_latest_log()
