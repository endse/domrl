import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_gallery():
    if not os.path.exists('logs/granular_history.csv'):
        print("Data file not found.")
        return
        
    df = pd.read_csv('logs/granular_history.csv')
    os.makedirs('logs/gallery', exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print("Generating Feature Distributions...")
    # 1-8. Individual Attribute Distplots
    cols = ['Enthusiasm', 'TimeOfDay', 'ScrollVel', 'Hover', 'ViewTime', 'w_Eng', 'w_Sat', 'w_Div', 'w_Fair']
    for col in cols:
        if col not in df.columns: continue
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.savefig(f'logs/gallery/dist_{col}.png')
        plt.close()
        
    print("Generating Action Analysis...")
    # 9. Action Count
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Action', palette='viridis')
    plt.title('Action Distribution (Overall)')
    plt.savefig('logs/gallery/action_counts.png')
    plt.close()
    
    # 10. Action vs Scenario
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Action', hue='Scenario', palette='Set2')
    plt.title('Action Choice by Scenario')
    plt.savefig('logs/gallery/action_by_scenario.png')
    plt.close()
    
    # 11. Action Heatmap (Time vs Action)
    # Bin time into hours
    df['TimeBin'] = df['TimeOfDay'].astype(int)
    pivot = df.pivot_table(index='Action', columns='TimeBin', values='Reward', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap='coolwarm', annot=False)
    plt.title('Average Reward Heatmap: Action vs Hour of Day')
    plt.savefig('logs/gallery/heatmap_action_time.png')
    plt.close()
    
    # 12-15. Variable Relations (Scatter/Hex)
    relations = [
        ('ScrollVel', 'Satisfaction'),
        ('Hover', 'Satisfaction'),
        ('Enthusiasm', 'Reward'),
        ('ScrollVel', 'Reward')
    ]
    for x, y in relations:
        plt.figure(figsize=(8, 6))
        sns.regplot(data=df.sample(min(1000, len(df))), x=x, y=y, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        plt.title(f'Correlation: {x} vs {y}')
        plt.savefig(f'logs/gallery/scatter_{x}_{y}.png')
        plt.close()
        
    print("Generating Episode Traces...")
    # 16-35. Trace Analysis for 5 random episodes
    # Pick 5 long episodes
    ep_ids = df[df['Step'] > 20]['Episode'].unique()
    if len(ep_ids) > 0:
        selected_eps = np.random.choice(ep_ids, min(5, len(ep_ids)), replace=False)
        
        for i, ep in enumerate(selected_eps):
            ep_data = df[df['Episode'] == ep]
            
            # 4 plots per episode
            fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            fig.suptitle(f'Episode {ep} Trace (Scenario: {ep_data["Scenario"].iloc[0]})')
            
            axs[0].plot(ep_data['Step'], ep_data['Satisfaction'], color='green', marker='o')
            axs[0].set_ylabel('Satisfaction')
            
            axs[1].plot(ep_data['Step'], ep_data['Reward'], color='blue', linestyle='--')
            axs[1].set_ylabel('Reward')
            
            axs[2].bar(ep_data['Step'], ep_data['Action'], color='purple', alpha=0.5)
            axs[2].set_ylabel('Action (Cat)')
            
            axs[3].plot(ep_data['Step'], ep_data['ScrollVel'], color='orange')
            axs[3].set_ylabel('Scroll Velocity')
            axs[3].set_xlabel('Step')
            
            plt.tight_layout()
            plt.savefig(f'logs/gallery/trace_ep_{i+1}_{ep}.png')
            plt.close()

    print("Generating Weight Analysis...")
    # 36. 3D Weight Scatter (Projected to 2D pairwise)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x='w_Eng', y='w_Sat', hue='Scenario', size='w_Div')
    plt.title('Weight Distribution Landscape')
    plt.savefig('logs/gallery/weight_landscape.png')
    plt.close()
    
    # 37-39. Reward vs Weight Correlations
    for w in ['w_Eng', 'w_Sat', 'w_Div', 'w_Fair']:
        if w not in df.columns: continue
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df, x=df[w].round(1), y='Reward')
        plt.title(f'Avg Reward vs {w}')
        plt.savefig(f'logs/gallery/reward_vs_{w}.png')
        plt.close()
        
    # 40-50. Additional Statistical plots
    # Churn distribution
    churn_steps = df[df['Done'] == True]['Step']
    plt.figure(figsize=(8, 5))
    sns.histplot(churn_steps, bins=20, color='red')
    plt.title('Step Number where Episode Ends (Churn/Max)')
    plt.savefig('logs/gallery/churn_step_dist.png')
    plt.close()
    
    # Satisfaction moving avg over all training
    plt.figure(figsize=(12, 6))
    df['Sat_MA'] = df['Satisfaction'].rolling(window=1000).mean()
    sns.lineplot(data=df, x=df.index, y='Sat_MA')
    plt.title('Global Satisfaction Trend (Moving Avg)')
    plt.savefig('logs/gallery/global_sat_trend.png')
    plt.close()
    
    # Pairplot (Big one)
    print("Generating Pairplot (this may take a moment)...")
    subset = df[['Enthusiasm', 'ScrollVel', 'Satisfaction', 'Reward', 'Scenario']]
    pp = sns.pairplot(subset.sample(min(1000, len(df))), hue='Scenario')
    pp.savefig('logs/gallery/pairplot_overview.png')
    plt.close()
    
    # Action Confidence Proxy (Frequency of max action)
    action_freq = df.groupby('Episode')['Action'].agg(lambda x: x.value_counts(normalize=True).max())
    plt.figure(figsize=(8, 5))
    sns.histplot(action_freq, color='magenta')
    plt.title('Action Dominance (Did agent spam one action?)')
    plt.savefig('logs/gallery/action_dominance.png')
    plt.close()
    
    # ... Add a few more specific Action breakdowns
    for act in range(5): # First 5 categories
        plt.figure(figsize=(6, 4))
        sub = df[df['Action'] == act]
        if not sub.empty:
            sns.kdeplot(data=sub, x='TimeOfDay', fill=True)
            plt.title(f'Time Usage for Action {act}')
            plt.savefig(f'logs/gallery/action_{act}_time_dist.png')
        plt.close()
        
    print("Gallery Generation Complete.")

if __name__ == "__main__":
    generate_gallery()
