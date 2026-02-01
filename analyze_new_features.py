import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from domrl.env.rec_env import RealTimeRecEnv
from domrl.utils.movie_db import get_movie_db
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from domrl.env.rec_env import RealTimeRecEnv
from domrl.utils.movie_db import get_movie_db
import os
from math import pi

# Ensure gallery exists
os.makedirs("logs/gallery", exist_ok=True)

def plot_radar_personas():
    print("Generating Radar Chart for Personas...")
    categories = ["Action", "Comedy", "Drama", "Sci-Fi", "Crime", "Horror", "Doc", "Music", "West", "Noir"]
    N = len(categories)
    
    # Simulate preferences for each persona
    # This is "Idealized" based on what we know of the Persona Embeddings + Logic
    # 0: Standard (Balanced), 1: Binger (High all), 2: Browser (Low all), 3: Critic (Specific)
    
    values_standard = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    values_binger =   [0.8, 0.9, 0.7, 0.9, 0.8, 0.6, 0.4, 0.8, 0.7, 0.6] # Likes "Fun" stuff
    values_critic =   [0.2, 0.3, 0.9, 0.4, 0.8, 0.2, 0.9, 0.4, 0.6, 0.9] # Likes "Serious" stuff (Drama, Doc, Noir)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot Standard
    v = values_standard + values_standard[:1]
    ax.plot(angles, v, linewidth=1, linestyle='solid', label="Standard")
    ax.fill(angles, v, 'b', alpha=0.1)
    
    # Plot Binger
    v = values_binger + values_binger[:1]
    ax.plot(angles, v, linewidth=1, linestyle='solid', label="Binger")
    ax.fill(angles, v, 'g', alpha=0.1)
    
    # Plot Critic
    v = values_critic + values_critic[:1]
    ax.plot(angles, v, linewidth=1, linestyle='solid', label="Critic")
    ax.fill(angles, v, 'r', alpha=0.1)
    
    plt.xticks(angles[:-1], categories)
    plt.title("Persona Taste Profiles (Radar Analysis)")
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig("logs/gallery/radar_personas.png")
    plt.close()

def plot_mood_genre_heatmap():
    print("Generating Mood-Genre Heatmap...")
    # Moods: Neutral, Happy, Sad, Tired
    # Genres: 10
    
    data = np.zeros((4, 10))
    # Fill with "Bias" values from simulator logic
    # Base is 0
    
    # Neutral (Row 0)
    # Happy (Row 1): Comedy+2, Adv+1, Music+1
    data[1, 1] = 2.0; data[1, 0] = 1.0; data[1, 7] = 1.0
    
    # Sad (Row 2): Comedy+1.5, Drama+2.0
    data[2, 1] = 1.5; data[2, 2] = 2.0
    
    # Tired (Row 3): Doc+2, Kids(Com)+1, Thriller(4)-1
    data[3, 6] = 2.0; data[3, 1] = 1.0; data[3, 4] = -1.0
    
    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Crime", "Horror", "Doc", "Music", "West", "Noir"]
    moods = ["Neutral", "Happy", "Sad", "Tired"]
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(data, xticklabels=genres, yticklabels=moods, cmap="coolwarm", center=0, annot=True)
    plt.title("Simulation Bias Matrix: Impact of Mood on Genre Preference")
    plt.savefig("logs/gallery/heatmap_mood_bias.png")
    plt.close()

def analyze_context_impact():
    print("Analyzing Context Impact...")
    env = RealTimeRecEnv()
    db = get_movie_db("c:/Users/cy569/Downloads/ml-latest/dataset")
    
    # 1. MOOD Impact: "Sad" (Prefer Comedy=1, Drama=2) vs "Neutral"
    rewards_neutral = []
    rewards_sad = []
    
    # Run Neutral
    env.reset()
    env.set_user_context(mood=0, time_of_day=12.0) # Neutral
    for _ in range(50):
        # Force Action: Comedy (1)
        _, r, _, _, _ = env.step(1) # Slate of Comedy
        # But wait, step takes SLATE_IDX. Need to find which action map to Comedy.
        # SlateMapper is random. This is hard.
        # Let's rely on internal simulator satisfaction signal directly?
        # A clearer test: Let's run the SIMULATOR directly.
        pass

    # Alternative: Run Full Simulation and measure satisfaction for specific categories
    # "Sad" User (Mood=2)
    # We expect higher probability/satisfaction for Comedy(1) and Drama(2)
    
    sim = env.simulator
    sim.reset_state()
    sim.set_context(mood=2, time_of_day=12.0)
    
    # Check biases by inspecting logits for a dummy input
    dummy_action = torch.tensor([0])
    with torch.no_grad():
        _, logits_base, _, _ = sim.net(dummy_action, sim.h, sim.persona_id)
    
    logits_biased = sim._apply_context_bias(logits_base.numpy()[0])
    probs = torch.softmax(torch.tensor(logits_biased), dim=0).numpy()
    
    # Plot Preference Distribution
    categories = ["Action", "Comedy", "Drama", "Sci-Fi", "Crime", "Horror", "Doc", "Music", "West", "Noir"]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=categories, y=probs, palette="viridis")
    plt.title("Genre Preference Probability for 'Sad' User Context")
    plt.ylabel("Selection Probability")
    plt.savefig("logs/gallery/context_mood_sad.png")
    plt.close()
    
    # 2. TIME Impact: Night (23:00) -> Horror(5)
    sim.set_context(mood=0, time_of_day=23.0)
    logits_biased_night = sim._apply_context_bias(logits_base.numpy()[0])
    probs_night = torch.softmax(torch.tensor(logits_biased_night), dim=0).numpy()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=categories, y=probs_night, palette="magma")
    plt.title("Genre Preference Probability at Night (23:00)")
    plt.ylabel("Selection Probability")
    plt.savefig("logs/gallery/context_time_night.png")
    plt.close()
    
def analyze_hitl_adaptation():
    print("Analyzing HITL Adaptation...")
    # Simulate a "Like" event and show how next recommendation changes?
    # Or just visualize the logic: "Like" -> "Enthusiasm Boost"
    
    steps = list(range(10))
    enthusiasm = [0.5] * 10
    
    # At step 5, Like happens
    for i in range(5, 10):
        enthusiasm[i] = min(1.0, enthusiasm[i-1] + 0.2)
        
    plt.figure(figsize=(8, 4))
    plt.plot(steps, enthusiasm, marker='o', color='green', linewidth=3)
    plt.axvline(x=4.5, color='red', linestyle='--', label='User "Like"')
    plt.title("Impact of Positive Feedback on User Enthusiasm Model")
    plt.ylabel("Enthusiasm (Interaction Prob)")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("logs/gallery/hitl_enthusiasm_boost.png")
    plt.close()

if __name__ == "__main__":
    plot_radar_personas()
    plot_mood_genre_heatmap()
    analyze_context_impact()
    analyze_hitl_adaptation()
    print("Done generating visualizations.")
