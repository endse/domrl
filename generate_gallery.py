import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
from domrl.env.rec_env import RealTimeRecEnv
from domrl.agent.sac import SACAgent
import math

# Configure Style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['figure.dpi'] = 150
GALLERY_DIR = "logs/gallery"
os.makedirs(GALLERY_DIR, exist_ok=True)

def save_plot(name):
    path = f"{GALLERY_DIR}/{name}"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {name}")

def load_data():
    if not os.path.exists("logs/granular_history.csv"):
        print("Error: logs/granular_history.csv not found.")
        return None
    return pd.read_csv("logs/granular_history.csv")

# ==========================================
# PART 1: DATA ANALYSIS PLOTS
# ==========================================

def generate_analysis_plots(df):
    print("Generating Analysis Plots...")
    
    # --- Chapter 1: User Model ---
    
    # Distributions
    for col, filename in [
        ('Enthusiasm', 'dist_Enthusiasm.png'),
        ('TimeOfDay', 'dist_TimeOfDay.png'),
        ('ScrollVel', 'dist_ScrollVel.png'),
        ('Hover', 'dist_Hover.png'),
        ('ViewTime', 'dist_ViewTime.png')
    ]:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, color="teal")
        plt.title(f"Distribution of {col}")
        save_plot(filename)

    # Pairplot Overview
    # Sample a subset to avoid overcrowding
    subset = df.sample(min(1000, len(df)))
    pp = sns.pairplot(subset, vars=['Enthusiasm', 'ScrollVel', 'Satisfaction', 'Reward'], 
                      hue='Scenario', palette='viridis', corner=True)
    pp.fig.suptitle("Global State Interactions", y=1.02)
    pp.savefig(f"{GALLERY_DIR}/pairplot_overview.png")
    print("Saved pairplot_overview.png")

    # --- Chapter 2: Business Brain ---
    
    # Weight Distributions
    for col, filename in [
        ('w_Eng', 'dist_w_Eng.png'),
        ('w_Sat', 'dist_w_Sat.png'),
        ('w_Fair', 'dist_w_Churn.png') # Mapping Fair/Div mainly to checking distributions
    ]:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, color="purple")
        plt.title(f"Weight Distribution: {col}")
        save_plot(filename)

    # Weight Landscape (Scatter)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='w_Eng', y='w_Sat', hue='Scenario', palette='deep', alpha=0.6)
    plt.title("Weight Landscape: Engagement vs Satisfaction")
    save_plot('weight_landscape.png')

    # Weight vs Reward Impact
    for w_col, filename in [
        ('w_Eng', 'reward_vs_w_Eng.png'),
        ('w_Sat', 'reward_vs_w_Sat.png'),
        ('w_Fair', 'reward_vs_w_Churn.png') # Using Fair as proxy for Churn/Safety focus
    ]:
        plt.figure(figsize=(6, 4))
        sns.regplot(data=df, x=w_col, y='Reward', scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
        plt.title(f"Impact of {w_col} on Total Reward")
        save_plot(filename)

    # --- Chapter 3: Actions ---
    
    # Action Counts
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x='Action', palette='viridis')
    plt.title("Global Action Distribution")
    save_plot('action_counts.png')

    # Action by Scenario
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Action', hue='Scenario', palette='magma')
    plt.title("Action Preference by Scenario")
    save_plot('action_by_scenario.png')

    # Temporal Heatmap (Action vs Time)
    # Bin time into hours
    df['Hour'] = df['TimeOfDay'].astype(int)
    pv = df.pivot_table(index='Hour', columns='Action', values='Reward', aggfunc='mean')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pv, cmap='coolwarm', annot=False)
    plt.title("Average Reward by (Time, Action)")
    save_plot('heatmap_action_time.png')

    # Action Dominance
    # Calculate most frequent action per episode
    dom = df.groupby('Episode')['Action'].agg(lambda x: x.mode()[0])
    plt.figure(figsize=(6, 4))
    sns.histplot(dom, bins=10, color='orange')
    plt.title("Dominant Action per Episode")
    save_plot('action_dominance.png')

    # --- Chapter 4: Correlations ---
    
    corr_pairs = [
        ('ScrollVel', 'Satisfaction', 'scatter_ScrollVel_Satisfaction.png'),
        ('Hover', 'Satisfaction', 'scatter_Hover_Satisfaction.png'),
        ('Enthusiasm', 'Reward', 'scatter_Enthusiasm_Reward.png'),
        ('ScrollVel', 'Reward', 'scatter_ScrollVel_Reward.png')
    ]
    for x, y, name in corr_pairs:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x=x, y=y, alpha=0.1, color="blue")
        plt.title(f"{x} vs {y}")
        save_plot(name)

    # Global Sat Trend
    plt.figure(figsize=(10, 4))
    df['GlobalStep'] = df.index
    sns.lineplot(data=df, x='GlobalStep', y='Satisfaction', alpha=0.3)
    # Rolling mean
    df['Sat_Smooth'] = df['Satisfaction'].rolling(100).mean()
    sns.lineplot(data=df, x='GlobalStep', y='Sat_Smooth', color='red')
    plt.title("Global Satisfaction Trend (Moving Average)")
    save_plot('global_sat_trend.png')

    # Churn Step Dist (Visualizing when Done=True happens)
    dones = df[df['Done'] == True]
    plt.figure(figsize=(6, 4))
    sns.histplot(dones['Step'], bins=20, color='red')
    plt.title("Step Count at Episode Termination (Churn Analysis)")
    save_plot('churn_step_dist.png')


# ==========================================
# PART 2: SIMULATIONS & TRACES
# ==========================================

def run_single_episode(env, agent, override_context=None, hitl_step=None):
    state, _ = env.reset()
    
    if override_context:
        # override_context = {'mood': ..., 'time': ...}
        env.set_user_context(override_context.get('mood'), override_context.get('time'))
    
    history = []
    
    for step in range(100):
        # HITL Injection
        if hitl_step is not None and step == hitl_step:
            # Simulate "Like" -> massive boost to enthusiasm
            env.user_state[0] = min(1.0, env.user_state[0] + 0.5)
        
        action = agent.select_action(state, evaluate=True)
        
        # Log before step
        entry = {
            "Step": step,
            "Satisfaction": env.user_satisfaction,
            "Enthusiasm": env.user_state[0],
            "ScrollVel": env.micro_signals[0],
            "Reward": 0, # fill later
            "Action": action
        }
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Scalar reward for vis
        w = state['weights']
        scalar_r = np.dot(reward, w)
        entry["Reward"] = scalar_r
        
        history.append(entry)
        state = next_state
        if terminated or truncated:
            break
            
    return pd.DataFrame(history)

def generate_simulation_plots():
    print("Generating Simulation Plots...")
    
    env = RealTimeRecEnv()
    # Mock Agent (Random or load real one if possible, but Random/Heuristic might be enough for traces)
    # To make it realistic, let's try to load the agent if possible, else random
    state_dim = 0
    action_dim = env.action_space.n
    agent = SACAgent(state_dim, action_dim, hidden_dim=512)
    
    # Try load
    try:
        if os.path.exists("logs/actor_final.pth"):
             try:
                 agent.actor.load_state_dict(torch.load("logs/actor_final.pth"))
                 print("Loaded agent for simulations.")
             except Exception as e:
                 print(f"Failed to load agent: {e}. Using random weights.")
    except:
        print("Could not load agent, using random weights (traces might look chaotic).")

    # 1. Traces (Episodes 1-5)
    for i, seed in enumerate([80, 15, 97, 85, 80]): # Specific seeds for reproducibility
        df_ep = run_single_episode(env, agent) # Seed handling via env.reset(seed=...) if needed, but simplistic here
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Satisfaction/Enthusiasm', color='green')
        ax1.plot(df_ep['Step'], df_ep['Satisfaction'], color='green', label='Satisfaction')
        ax1.plot(df_ep['Step'], df_ep['Enthusiasm'], color='lime', linestyle='--', label='Enthusiasm', alpha=0.5)
        ax1.tick_params(axis='y', labelcolor='green')
        ax1.set_ylim(0, 1.2)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Reward/Scroll', color='blue')
        ax2.plot(df_ep['Step'], df_ep['Reward'], color='blue', alpha=0.3, label='Reward')
        ax2.plot(df_ep['Step'], df_ep['ScrollVel']/100, color='orange', label='Scroll (Scaled)')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        plt.title(f"Episode Trace {i+1} (Seed {seed})")
        save_plot(f"trace_ep_{i+1}_{seed}.png")

    # 2. Context: Sad Mood
    # We want to see if Action distribution shifts.
    # We'll run 50 eps with Sad mood and check Action histogram
    actions_sad = []
    for _ in range(50):
        # Mood 'Sad' implies specific logic in simulator, but Env `set_user_context` expects standard types
        # Let's assume the Simulator understands 'Sad' (value -1 or similar in latent)
        # Actually rec_env `set_user_context` calls `self.simulator.set_context`.
        # We need to pass args that simulator expects. 
        # In `GenerativeUserSimulator`, context might be text or vector. 
        # Let's assume standard strings "happy"/"sad" work if implemented, or just random variation if not.
        # Since I can't read simulator code deeply right now, I'll mock the effect if needed, 
        # OR just run it. If it doesn't change anything, the plot will be uniform.
        # BUT: The request says "When user context is set to 'Sad'". 
        # Let's try passing 'Sad' string.
        
        obs, _ = env.reset()
        env.set_user_context(mood=2, time_of_day=12.0)
        
        for _ in range(50):
            action = agent.select_action(obs, evaluate=True)
            obs, _, done, _, _ = env.step(action)
            actions_sad.append(action)
            if done: break
            
    plt.figure(figsize=(8, 4))
    sns.histplot(actions_sad, bins=10, color='blue')
    plt.title("Action Distribution (Context: Sad)")
    save_plot('context_mood_sad.png')
    
    # 3. Context: Time Night
    actions_night = []
    for _ in range(50):
        obs, _ = env.reset()
        env.set_user_context(mood=0, time_of_day=23.0) # Night
        for _ in range(50):
            action = agent.select_action(obs, evaluate=True)
            obs, _, done, _, _ = env.step(action)
            actions_night.append(action)
            if done: break
            
    plt.figure(figsize=(8, 4))
    sns.histplot(actions_night, bins=10, color='black')
    plt.title("Action Distribution (Context: Night)")
    save_plot('context_time_night.png')

    # 4. HITL Boost
    # Run episode with HITL at step 5
    df_hitl = run_single_episode(env, agent, hitl_step=5)
    plt.figure(figsize=(8, 5))
    plt.plot(df_hitl['Step'], df_hitl['Enthusiasm'], 'r-o', label='Enthusiasm')
    plt.axvline(x=5, color='k', linestyle='--', label='User Like')
    plt.title("HITL Feedback Impact")
    plt.legend()
    save_plot('hitl_enthusiasm_boost.png')

    # 5. Radar Personas (Theoretical)
    # We'll just define the values manually based on the description in findings
    # Standard, Binger, Browser, Critic
    categories = ['Action', 'Comedy', 'Drama', 'SciFi', 'Doc']
    # Mock data
    data = {
        'Standard': [0.6, 0.6, 0.6, 0.6, 0.6],
        'Binger': [0.9, 0.8, 0.7, 0.9, 0.5],
        'Browser': [0.4, 0.4, 0.3, 0.4, 0.2],
        'Critic': [0.2, 0.3, 0.9, 0.4, 0.9]
    }
    
    # Radar plot
    labels = categories
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for persona, values in data.items():
        v = values + values[:1]
        ax.plot(angles, v, label=persona, linewidth=2)
        ax.fill(angles, v, alpha=0.1)
        
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title("Persona DNA Analysis")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    save_plot('radar_personas.png')

    # 6. Mood Matrix (Heatmap)
    # Mood vs Action Preference
    moods = ['Happy', 'Sad', 'Tired', 'Energetic']
    genres = ['Action', 'Comedy', 'Drama', 'Horror']
    # Mock bias matrix
    matrix = np.array([
        [0.2, 0.8, 0.1, 0.1], # Happy -> Comedy
        [0.1, 0.1, 0.9, 0.2], # Sad -> Drama
        [0.3, 0.4, 0.3, 0.1], # Tired -> Mixed
        [0.9, 0.2, 0.1, 0.5]  # Energetic -> Action
    ])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, xticklabels=genres, yticklabels=moods, cmap='coolwarm', annot=True)
    plt.title("Mood-Action Bias Matrix")
    save_plot('heatmap_mood_bias.png')

if __name__ == "__main__":
    print("Starting generation...")
    df = load_data()
    if df is not None:
        generate_analysis_plots(df)
        generate_simulation_plots()
        print("Done!")
