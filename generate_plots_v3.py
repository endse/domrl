"""
DOM-RL v3.0 — Plot Generation Script

Generates all visualizations for the updated finds.md documentation:
- 5-objective reward analysis
- NSGA-II Pareto front visualization
- Micro-behavioral signal distributions (scroll velocity, hover-dwell, skip gradient)
- Cold start persona inference analysis
- Hybrid SAC-NSGA-II weight dynamics
- Episode traces with enriched signals
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

# Suppress torch warnings
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "logs/gallery_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'figure.figsize': (10, 6),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'engagement': '#58a6ff',
    'satisfaction': '#3fb950',
    'diversity': '#d2a8ff',
    'fairness': '#f0883e',
    'churn': '#f85149',
    'scroll': '#79c0ff',
    'hover': '#56d364',
    'skip': '#ffa657',
    'pareto': '#ff7b72',
    'cold': '#a5d6ff',
}


def collect_episode_data(num_episodes=30, max_steps=50):
    """Run episodes and collect data for analysis."""
    from domrl.env.rec_env import RealTimeRecEnv
    from domrl.agent.sac import SACAgent
    from domrl.agent.weight_agent import WeightAgent
    
    env = RealTimeRecEnv()
    action_dim = env.action_space.shape[0]
    sac = SACAgent(0, action_dim, num_items=env.num_categories, hidden_dim=128)
    wa = WeightAgent(action_dim=action_dim, num_items=env.num_categories,
                     num_objectives=env.num_objectives, nsga2_pop_size=20, nsga2_generations=5)
    
    all_data = {
        'rewards': [], 'satisfaction': [], 'churn_prob': [],
        'scroll_velocity': [], 'hover_dwell_ratio': [], 'skip_gradient': [],
        'scroll_raw': [], 'hover_raw': [], 'view_time': [],
        'weights': [], 'reward_vectors': [],
        'ep_rewards': [], 'ep_satisfaction': [], 'ep_lengths': [],
        'persona': [], 'cold_start': [],
        'traces': [],  # Per-episode traces
    }
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        trace = {'rewards': [], 'sat': [], 'scroll': [], 'hover': [], 'skip': [], 'actions': [], 'weights': []}
        
        for step in range(max_steps):
            weights = wa.select_weights(obs)
            env.weights = weights
            obs['weights'] = weights
            
            action = sac.select_action(obs) if ep > 3 else env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            rv = info['reward_vector']
            wa.record_reward_vector(rv)
            
            signals = info['signals']
            all_data['scroll_velocity'].append(signals.get('scroll_velocity', 0))
            all_data['hover_dwell_ratio'].append(signals.get('hover_dwell_ratio', 0))
            all_data['skip_gradient'].append(signals.get('skip_gradient', 0))
            all_data['scroll_raw'].append(signals.get('scroll', 0))
            all_data['hover_raw'].append(signals.get('hover', 0))
            all_data['view_time'].append(signals.get('view_time', 0))
            all_data['rewards'].append(reward)
            all_data['satisfaction'].append(env.user_satisfaction)
            all_data['churn_prob'].append(info.get('churn_probability', 0))
            all_data['weights'].append(weights.copy())
            all_data['reward_vectors'].append(rv.copy())
            all_data['persona'].append(env.current_persona_id)
            all_data['cold_start'].append(signals.get('is_cold_start', False))
            
            trace['rewards'].append(reward)
            trace['sat'].append(env.user_satisfaction)
            trace['scroll'].append(signals.get('scroll_velocity', 0))
            trace['hover'].append(signals.get('hover_dwell_ratio', 0))
            trace['skip'].append(signals.get('skip_gradient', 0))
            trace['actions'].append(info.get('chosen_cat', 0))
            trace['weights'].append(weights.copy())
            
            ep_reward += reward
            obs = next_obs
            if done: break
        
        all_data['ep_rewards'].append(ep_reward)
        all_data['ep_satisfaction'].append(env.user_satisfaction)
        all_data['ep_lengths'].append(step + 1)
        all_data['traces'].append(trace)
        
        # Evolve NSGA-II periodically
        if (ep + 1) % 5 == 0:
            wa.evolve_nsga2()
    
    return all_data, wa


def plot_reward_composition(data):
    """5-objective reward vector breakdown - pie and bar."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    rv = np.array(data['reward_vectors'])
    avg_rv = np.mean(np.abs(rv), axis=0)
    labels = ['Engagement', 'Satisfaction', 'Diversity', 'Fairness', 'Churn\nMitigation']
    colors = [COLORS['engagement'], COLORS['satisfaction'], COLORS['diversity'], 
              COLORS['fairness'], COLORS['churn']]
    
    # Pie
    wedges, texts, autotexts = axes[0].pie(avg_rv, labels=labels, colors=colors, autopct='%1.1f%%',
                                           textprops={'color': '#c9d1d9', 'fontsize': 9})
    axes[0].set_title('5-Objective Reward Composition\n(Paper III-C)', fontsize=13, fontweight='bold')
    
    # Bar
    x = np.arange(len(labels))
    bars = axes[1].bar(x, avg_rv, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel('Mean |Reward|')
    axes[1].set_title('Per-Objective Mean Reward Magnitude', fontsize=13, fontweight='bold')
    
    for bar, val in zip(bars, avg_rv):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='#c9d1d9')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/reward_composition_5obj.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] reward_composition_5obj.png")


def plot_micro_behavioral_signals(data):
    """Paper III-A: Enriched micro-behavioral signal distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Scroll Velocity 
    axes[0].hist(data['scroll_velocity'], bins=40, color=COLORS['scroll'], alpha=0.8, edgecolor='white', linewidth=0.3)
    axes[0].axvline(np.mean(data['scroll_velocity']), color='white', linestyle='--', linewidth=1.5, label=f"Mean: {np.mean(data['scroll_velocity']):.2f}")
    axes[0].set_title('Scroll Velocity Distribution\n(Paper III-A: High = High-Entropy Search)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Velocity')
    axes[0].legend()
    
    # Hover-Dwell Ratio
    axes[1].hist(data['hover_dwell_ratio'], bins=40, color=COLORS['hover'], alpha=0.8, edgecolor='white', linewidth=0.3)
    axes[1].axvline(np.mean(data['hover_dwell_ratio']), color='white', linestyle='--', linewidth=1.5, label=f"Mean: {np.mean(data['hover_dwell_ratio']):.2f}")
    axes[1].set_title('Hover-Dwell Ratio Distribution\n(Paper III-A: Card Dwell vs Session)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Ratio')
    axes[1].legend()
    
    # Skip-Rate Gradient
    skip_data = data['skip_gradient']
    axes[2].hist(skip_data, bins=40, color=COLORS['skip'], alpha=0.8, edgecolor='white', linewidth=0.3)
    axes[2].axvline(np.mean(skip_data), color='white', linestyle='--', linewidth=1.5, label=f"Mean: {np.mean(skip_data):.2f}")
    axes[2].axvline(-1.0, color=COLORS['churn'], linestyle=':', linewidth=1.5, alpha=0.7, label='Early Skip (<3s)')
    axes[2].axvline(-0.1, color=COLORS['satisfaction'], linestyle=':', linewidth=1.5, alpha=0.7, label='Late Skip (>12s)')
    axes[2].set_title('Skip-Rate Temporal Gradient\n(Paper III-A: -1=Titular Fail, -0.1=Quality Fail)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Gradient')
    axes[2].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/micro_behavioral_signals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] micro_behavioral_signals.png")


def plot_signal_correlations(data):
    """Correlation between micro-behavioral signals and satisfaction/reward."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    sat = np.array(data['satisfaction'])
    rew = np.array(data['rewards'])
    scroll = np.array(data['scroll_velocity'])
    hover = np.array(data['hover_dwell_ratio'])
    skip = np.array(data['skip_gradient'])
    
    pairs = [
        (scroll, sat, 'Scroll Velocity', 'Satisfaction', COLORS['scroll']),
        (hover, sat, 'Hover-Dwell Ratio', 'Satisfaction', COLORS['hover']),
        (skip, sat, 'Skip Gradient', 'Satisfaction', COLORS['skip']),
        (scroll, rew, 'Scroll Velocity', 'Reward', COLORS['scroll']),
        (hover, rew, 'Hover-Dwell Ratio', 'Reward', COLORS['hover']),
        (skip, rew, 'Skip Gradient', 'Reward', COLORS['skip']),
    ]
    
    for ax, (x, y, xlabel, ylabel, color) in zip(axes.flat, pairs):
        ax.scatter(x, y, alpha=0.3, s=8, c=color)
        # Trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, p(xline), color='white', linewidth=2, linestyle='--', alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        corr = np.corrcoef(x, y)[0, 1]
        ax.set_title(f'{xlabel} vs {ylabel}\nr={corr:.3f}', fontsize=10, fontweight='bold')
    
    plt.suptitle('Micro-Behavioral Signal Correlations (Paper III-A)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/signal_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] signal_correlations.png")


def plot_nsga2_pareto(wa):
    """Paper IV-D: NSGA-II Pareto front visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    obj = wa.nsga2.objectives
    pop = wa.nsga2.population
    fronts = []
    try:
        from domrl.agent.nsga2 import non_dominated_sort
        fronts = non_dominated_sort(obj)
    except:
        pass
    
    obj_names = ['Engagement', 'Satisfaction', 'Diversity', 'Fairness', 'Churn Mitigation']
    
    # 2D projections of Pareto front
    proj_pairs = [(0, 1), (0, 2), (1, 4)]
    for ax, (i, j) in zip(axes, proj_pairs):
        # All points
        ax.scatter(obj[:, i], obj[:, j], alpha=0.3, s=15, c='#8b949e', label='Population')
        
        # Pareto front
        if fronts:
            pf = fronts[0]
            ax.scatter(obj[pf, i], obj[pf, j], c=COLORS['pareto'], s=40, 
                      edgecolors='white', linewidth=0.5, zorder=5, label='Pareto Front')
        
        ax.set_xlabel(obj_names[i])
        ax.set_ylabel(obj_names[j])
        ax.set_title(f'{obj_names[i]} vs {obj_names[j]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
    
    plt.suptitle('NSGA-II Pareto Front Projections (Paper IV-D)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/nsga2_pareto_front.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] nsga2_pareto_front.png")


def plot_nsga2_weight_population(wa):
    """Visualize the NSGA-II population of weight vectors."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pop = wa.nsga2.population
    obj_names = ['Engage', 'Satisfy', 'Diverse', 'Fair', 'Anti-Churn']
    colors = [COLORS['engagement'], COLORS['satisfaction'], COLORS['diversity'], 
              COLORS['fairness'], COLORS['churn']]
    
    x = np.arange(len(obj_names))
    width = 0.7 / pop.shape[0]
    
    # Plot each individual as a thin bar group
    for i in range(min(pop.shape[0], 20)):  # Show max 20
        alpha = 0.3 + 0.7 * (i == 0)  # Highlight first (Pareto best)
        for j, color in enumerate(colors):
            ax.bar(x[j] + i * width - 0.35, pop[i, j], width, alpha=alpha, color=color, edgecolor='none')
    
    # Best balanced overlay
    best = wa.nsga2.get_best_weights("balanced")
    bars = ax.bar(x + 0.4, best, 0.08, color='white', edgecolor=COLORS['pareto'], linewidth=2, label='Best Balanced', zorder=10)
    for bar, val in zip(bars, best):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}',
               ha='center', fontsize=9, color=COLORS['pareto'], fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(obj_names)
    ax.set_ylabel('Weight Value')
    ax.set_title('NSGA-II Weight Population (Paper IV-D)\nEach bar group = one candidate weight vector', fontsize=13, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/nsga2_weight_population.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] nsga2_weight_population.png")


def plot_weight_dynamics(data):
    """Paper IV-E: Hybrid SAC-NSGA-II weight dynamics over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    weights = np.array(data['weights'])
    obj_names = ['Engagement', 'Satisfaction', 'Diversity', 'Fairness', 'Churn Mitigation']
    colors = [COLORS['engagement'], COLORS['satisfaction'], COLORS['diversity'], 
              COLORS['fairness'], COLORS['churn']]
    
    # Smooth weights
    window = 20
    for i, (name, color) in enumerate(zip(obj_names, colors)):
        smoothed = np.convolve(weights[:, i], np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, color=color, alpha=0.85, linewidth=1.5, label=name)
    
    axes[0].set_ylabel('Weight Value')
    axes[0].set_title('Dynamic Weight Adaptation (Paper IV-E: Hybrid SAC-NSGA-II)', fontsize=13, fontweight='bold')
    axes[0].legend(ncol=5, fontsize=8, loc='upper center')
    
    # Reward vectors over time
    rv = np.array(data['reward_vectors'])
    for i, (name, color) in enumerate(zip(obj_names, colors)):
        smoothed = np.convolve(rv[:, i], np.ones(window)/window, mode='valid')
        axes[1].plot(smoothed, color=color, alpha=0.85, linewidth=1.5, label=name)
    
    axes[1].set_ylabel('Objective Reward')
    axes[1].set_xlabel('Step')
    axes[1].set_title('Per-Objective Reward Dynamics', fontsize=13, fontweight='bold')
    axes[1].legend(ncol=5, fontsize=8, loc='upper center')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/weight_dynamics_hybrid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] weight_dynamics_hybrid.png")


def plot_cold_start_analysis(data):
    """Paper III (Challenge C): Cold Start persona inference analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    cold_mask = np.array(data['cold_start'], dtype=bool)
    warm_mask = ~cold_mask
    
    scroll = np.array(data['scroll_velocity'])
    hover = np.array(data['hover_dwell_ratio'])
    personas = np.array(data['persona'])
    
    # Cold vs Warm scroll velocity
    if np.any(cold_mask) and np.any(warm_mask):
        axes[0].hist(scroll[cold_mask], bins=25, alpha=0.7, color=COLORS['cold'], label=f'Cold Start (n={cold_mask.sum()})', density=True)
        axes[0].hist(scroll[warm_mask], bins=25, alpha=0.5, color=COLORS['engagement'], label=f'Warm (n={warm_mask.sum()})', density=True)
    else:
        axes[0].hist(scroll, bins=25, alpha=0.7, color=COLORS['cold'], label='All', density=True)
    axes[0].set_title('Scroll Velocity: Cold vs Warm\n(Paper III Challenge C)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Scroll Velocity')
    axes[0].legend(fontsize=8)
    
    # Cold vs Warm hover
    if np.any(cold_mask) and np.any(warm_mask):
        axes[1].hist(hover[cold_mask], bins=25, alpha=0.7, color=COLORS['cold'], label='Cold Start', density=True)
        axes[1].hist(hover[warm_mask], bins=25, alpha=0.5, color=COLORS['hover'], label='Warm', density=True)
    else:
        axes[1].hist(hover, bins=25, alpha=0.7, color=COLORS['cold'], label='All', density=True)
    axes[1].set_title('Hover-Dwell Ratio: Cold vs Warm', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Hover-Dwell Ratio')
    axes[1].legend(fontsize=8)
    
    # Persona distribution
    persona_names = ['Standard', 'Binger', 'Browser', 'Critic']
    persona_colors = [COLORS['engagement'], COLORS['satisfaction'], COLORS['diversity'], COLORS['churn']]
    counts = [np.sum(personas == i) for i in range(4)]
    bars = axes[2].bar(persona_names, counts, color=persona_colors, edgecolor='white', linewidth=0.5)
    for bar, count in zip(bars, counts):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', fontsize=10, color='#c9d1d9')
    axes[2].set_title('Persona Distribution Across Episodes', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cold_start_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] cold_start_analysis.png")


def plot_churn_dynamics(data):
    """Paper III-C: Churn probability and mitigation analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    churn = np.array(data['churn_prob'])
    sat = np.array(data['satisfaction'])
    
    # Churn probability distribution
    axes[0].hist(churn, bins=30, color=COLORS['churn'], alpha=0.8, edgecolor='white', linewidth=0.3)
    axes[0].axvline(np.mean(churn), color='white', linestyle='--', linewidth=1.5, label=f"Mean: {np.mean(churn):.3f}")
    axes[0].set_title('Churn Probability Distribution\n(Paper III-C: Inter-Session Interval)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Churn Probability')
    axes[0].legend()
    
    # Satisfaction vs Churn
    axes[1].scatter(sat, churn, alpha=0.3, s=8, c=COLORS['churn'])
    z = np.polyfit(sat, churn, 1)
    p = np.poly1d(z)
    xline = np.linspace(sat.min(), sat.max(), 100)
    axes[1].plot(xline, p(xline), color='white', linewidth=2, linestyle='--')
    corr = np.corrcoef(sat, churn)[0, 1]
    axes[1].set_title(f'Satisfaction vs Churn Probability\nr={corr:.3f}', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Satisfaction')
    axes[1].set_ylabel('Churn Probability')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/churn_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] churn_dynamics.png")


def plot_episode_traces(data, num_traces=3):
    """Detailed episode traces with enriched micro-behavioral signals."""
    for t_idx in range(min(num_traces, len(data['traces']))):
        trace = data['traces'][t_idx]
        steps = range(len(trace['rewards']))
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        # Panel 1: Reward + Satisfaction
        axes[0].plot(steps, trace['rewards'], color=COLORS['engagement'], alpha=0.8, linewidth=1.5, label='Reward')
        ax2 = axes[0].twinx()
        ax2.plot(steps, trace['sat'], color=COLORS['satisfaction'], alpha=0.8, linewidth=2, label='Satisfaction')
        ax2.set_ylabel('Satisfaction', color=COLORS['satisfaction'])
        axes[0].set_ylabel('Reward', color=COLORS['engagement'])
        axes[0].set_title(f'Episode {t_idx+1} Trace — Reward & Satisfaction', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
        
        # Panel 2: Micro-behavioral signals
        axes[1].plot(steps, trace['scroll'], color=COLORS['scroll'], alpha=0.8, linewidth=1.5, label='Scroll Velocity')
        axes[1].plot(steps, trace['hover'], color=COLORS['hover'], alpha=0.8, linewidth=1.5, label='Hover-Dwell Ratio')
        axes[1].plot(steps, trace['skip'], color=COLORS['skip'], alpha=0.8, linewidth=1.5, label='Skip Gradient')
        axes[1].set_ylabel('Signal Value')
        axes[1].set_title('Micro-Behavioral Signals (Paper III-A)', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=8)
        
        # Panel 3: Actions (Chosen Category)
        axes[2].bar(steps, trace['actions'], color=COLORS['diversity'], alpha=0.6, edgecolor='none')
        axes[2].set_ylabel('Category ID')
        axes[2].set_title('Chosen Content Category', fontsize=12, fontweight='bold')
        
        # Panel 4: Weight dynamics
        w = np.array(trace['weights'])
        obj_names = ['Engage', 'Satisfy', 'Diverse', 'Fair', 'Anti-Churn']
        w_colors = [COLORS['engagement'], COLORS['satisfaction'], COLORS['diversity'], COLORS['fairness'], COLORS['churn']]
        for i, (name, color) in enumerate(zip(obj_names, w_colors)):
            axes[3].plot(steps, w[:, i], color=color, alpha=0.8, linewidth=1.5, label=name)
        axes[3].set_ylabel('Weight')
        axes[3].set_xlabel('Step')
        axes[3].set_title('Objective Weight Dynamics (Paper IV-E)', fontsize=12, fontweight='bold')
        axes[3].legend(ncol=5, fontsize=7)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/episode_trace_{t_idx+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] episode_trace_{t_idx+1}.png")


def plot_training_summary(data):
    """Overall training summary: rewards, satisfaction, accuracy over episodes."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    eps = range(len(data['ep_rewards']))
    
    # Panel 1: Episode Rewards
    axes[0].bar(eps, data['ep_rewards'], color=COLORS['engagement'], alpha=0.6, edgecolor='none')
    window = max(1, len(data['ep_rewards']) // 5)
    if len(data['ep_rewards']) > window:
        smoothed = np.convolve(data['ep_rewards'], np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(data['ep_rewards'])), smoothed, color='white', linewidth=2, label='Moving Avg')
        axes[0].legend()
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('Training Summary — Episode Rewards', fontsize=13, fontweight='bold')
    
    # Panel 2: Satisfaction
    axes[1].plot(eps, data['ep_satisfaction'], color=COLORS['satisfaction'], alpha=0.8, linewidth=2, marker='o', markersize=4)
    axes[1].axhline(0.5, color='#8b949e', linestyle='--', alpha=0.5, label='Baseline (0.5)')
    axes[1].set_ylabel('Final Satisfaction')
    axes[1].set_title('Final Episode Satisfaction', fontsize=13, fontweight='bold')
    axes[1].legend()
    
    # Panel 3: Episode Length (Churn Indicator)
    colors_len = [COLORS['churn'] if l < 20 else COLORS['satisfaction'] for l in data['ep_lengths']]
    axes[2].bar(eps, data['ep_lengths'], color=colors_len, alpha=0.7, edgecolor='none')
    axes[2].axhline(50, color='#8b949e', linestyle='--', alpha=0.5, label='Max Steps')
    axes[2].set_ylabel('Episode Length')
    axes[2].set_xlabel('Episode')
    axes[2].set_title('Episode Length (Red = Early Churn)', fontsize=13, fontweight='bold')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] training_summary.png")


def plot_objective_radar(data):
    """Radar chart of average objective performance."""
    rv = np.array(data['reward_vectors'])
    avg = np.mean(rv, axis=0)
    
    labels = ['Engagement', 'Satisfaction', 'Diversity', 'Fairness', 'Churn\nMitigation']
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    # Normalize to 0-1 for radar
    mins = rv.min(axis=0)
    maxs = rv.max(axis=0)
    ranges = maxs - mins + 1e-10
    normalized = (avg - mins) / ranges
    values = normalized.tolist()
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor('#161b22')
    
    ax.plot(angles, values, color=COLORS['engagement'], linewidth=2)
    ax.fill(angles, values, color=COLORS['engagement'], alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title('5-Objective Performance Radar\n(Paper III-C)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/objective_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] objective_radar.png")


def plot_mood_bias_heatmap():
    """Mood/Time bias matrix from the simulator."""
    from domrl.env.user_simulator import GenerativeUserSimulator
    
    categories = ['Action', 'Comedy', 'Drama', 'SciFi', 'Crime', 'Horror', 'Doc', 'Music', 'West', 'Noir']
    moods = ['Neutral', 'Happy', 'Sad', 'Tired']
    
    bias_matrix = np.zeros((len(moods), len(categories)))
    
    for m_idx, mood in enumerate(moods):
        sim = GenerativeUserSimulator()
        sim.reset_state()
        sim.set_context(mood=m_idx, time_of_day=12.0)
        base_logits = np.zeros(10)
        biased = sim._apply_context_bias(base_logits)
        bias_matrix[m_idx] = biased
    
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(bias_matrix, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_yticks(range(len(moods)))
    ax.set_yticklabels(moods)
    
    for i in range(len(moods)):
        for j in range(len(categories)):
            val = bias_matrix[i, j]
            if abs(val) > 0.1:
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=9,
                       color='white' if abs(val) > 1.5 else '#c9d1d9')
    
    plt.colorbar(im, label='Logit Bias')
    ax.set_title('Mood-Context Bias Matrix (Paper III-A: Context Awareness)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/mood_bias_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] mood_bias_heatmap.png")


if __name__ == "__main__":
    print("=" * 60)
    print("DOM-RL v3.0 — Plot Generation")
    print("=" * 60)
    print()
    
    print("Collecting episode data (30 episodes x 50 steps)...")
    data, wa = collect_episode_data(num_episodes=30, max_steps=50)
    print(f"  Collected {len(data['rewards'])} total steps\n")
    
    print("Generating plots...")
    plot_reward_composition(data)
    plot_micro_behavioral_signals(data)
    plot_signal_correlations(data)
    plot_nsga2_pareto(wa)
    plot_nsga2_weight_population(wa)
    plot_weight_dynamics(data)
    plot_cold_start_analysis(data)
    plot_churn_dynamics(data)
    plot_episode_traces(data, num_traces=3)
    plot_training_summary(data)
    plot_objective_radar(data)
    plot_mood_bias_heatmap()
    
    print(f"\nAll plots saved to {OUTPUT_DIR}/")
    print("Done!")
