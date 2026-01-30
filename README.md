# DOM-RL

**Dynamic Multi-Objective Deep Reinforcement Learning for Real-Time Recommendation**

DOM-RL is an advanced recommendation framework that uses Hierarchical Reinforcement Learning to dynamically balance conflicting business objectives (Engagement, Satisfaction, Diversity, Fairness) in real-time.

It features a robust **Offline-to-Online** pipeline, a **Generative User Simulator** with realistic boredom dynamics, and a professional **Live Command Center** for inspection.

![Dashboard Preview](https://via.placeholder.com/800x400?text=DOM-RL+Enterprise+Dashboard)

## 🚀 Key Features

### 1. Enterprise Command Center
A professional Streamlit-based dashboard (`app.py`) for live interaction and inspection.
-   **Real-Time Confidence**: Displays the Policy Network's raw certainty (0-100%).
-   **Objective Override**: Manually force the agent to prioritize "Fairness" or "Diversity" to test its adaptability.
-   **Internal Diagnostics**: Visualize the **CVaR Risk Distribution** (Quantile Regression) and raw state vectors.
-   **Model Hot-Swapping**: Automatically detects and loads the latest training checkpoints without restarting.

### 2. Generative User Simulator v2
-   **Boredom Dynamics**: The simulator now models "feature fatigue". Repeatedly recommending the same category triggers a satisfaction penalty (churn risk).
-   **Diversity Bonus**: Users receive a dopamine boost (Satisfaction +) when presented with a diverse slate of unique items.
-   **Latent State GRU**: Models hidden intent evolution using a Gated Recurrent Unit.

### 3. Hierarchical & Distributional RL
-   **Weight Agent**: A Meta-Controller using **Distributional RL** to output dynamic objective weights $w$. It balances **Performance** vs **Risk** (CVaR).
-   **SAC Agent**: A Soft Actor-Critic agent (now scaled to 512 hidden units) that maps state + weights to the optimal **Slate** of recommendations.

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/) (Optional, for offline training)

### Setup
1. Clone the repository.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 🛠️ Usage

### 1. 🖥️ Launch the Enterprise Dashboard
Evaluate the agent in real-time using the interactive Command Center.
```powershell
streamlit run app.py
```
*   **Mission Control**: Use the sidebar to switch Personas (Critic, Binger) or enable Manual Overrides.
*   **Analytics Tab**: Watch Satisfaction trends and Weight adaptation in real-time.

### 2. 🧠 Training the Brain
Train the agent to adapt to the new "Boredom" dynamics.

**High-Accuracy CPU Run:**
Optimized for non-GPU environments (Batch Size 256, 2000 Episodes).
```powershell
python train.py --max_episodes 2000 --slate_size 3 --batch_size 256
```

**Hybrid Run (Offline + Online):**
Warm-start with MovieLens data before fine-tuning online.
```powershell
python train.py --dataset_path "C:/path/to/dataset" --cql_weight 1.0 --bc_weight 0.5
```

### 3. 📊 Analysis Tools
DOM-RL includes a suite of visualization scripts:
-   `collect_granular_data.py`: Generates detailed behavioral logs.
-   `generate_gallery.py`: Creates ~40 plots in `logs/gallery/` (State t-SNE, Weight Distributions, etc.).
-   `finds.md`: A comprehensive auto-generated technical report.

---

## 🧩 Project Structure

- `domrl/`
    - `agent/`: **SACAgent** (CQL+BC, 512 units) and **WeightAgent** (Distributional).
    - `env/`: `RealTimeRecEnv` and `GenerativeUserSimulator` (Boredom Logic).
    - `models/`: Network architectures (`StateEncoder`, `UserDynamicsNet`).
- `app.py`: **Enterprise Dashboard** source code.
- `train.py`: Main training loop with Hit Rate tracking.
- `logs/`: Checkpoints (`actor_*.pth`) and TensorBoard logs.

## License
MIT
