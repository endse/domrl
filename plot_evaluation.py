import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_scenarios():
    if not os.path.exists('logs/evaluation_results_hybrid.csv'):
        print("logs/evaluation_results_hybrid.csv not found. Run evaluate.py first.")
        return

    df = pd.read_csv('logs/evaluation_results_hybrid.csv')
    
    # Setup style
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Agent Behavior under Dynamic Objectives', fontsize=16)
    
    # 1. Average Reward
    sns.barplot(data=df, x='scenario', y='reward', ax=axs[0], palette="viridis", errorbar='sd')
    axs[0].set_title('Average Reward per Episode')
    axs[0].set_ylabel('Reward')
    axs[0].set_xlabel('')
    
    # 2. Churn Rate
    # Churn is binary 0/1, so mean is the rate
    sns.barplot(data=df, x='scenario', y='churned', ax=axs[1], palette="magma", errorbar=None)
    axs[1].set_title('Churn Rate (Risk of User Exit)')
    axs[1].set_ylabel('Churn Probability')
    axs[1].set_xlabel('')
    axs[1].set_ylim(0, 1.1)
    
    # 3. Satisfaction
    sns.boxplot(data=df, x='scenario', y='satisfaction', ax=axs[2], palette="coolwarm")
    axs[2].set_title('User Satisfaction Distribution')
    axs[2].set_ylabel('Satisfaction (0-1)')
    axs[2].set_xlabel('')
    
    plt.tight_layout()
    plt.savefig('logs/scenario_comparison.png')
    print("Saved plot to logs/scenario_comparison.png")

if __name__ == "__main__":
    plot_scenarios()
