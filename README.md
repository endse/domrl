# DOM-RL

**Dynamic Multi-Objective Deep Reinforcement Learning for Real-Time Recommendation**

DOM-RL is an advanced recommendation framework that uses Hierarchical Reinforcement Learning to dynamically balance conflicting business objectives (Engagement, Satisfaction, Diversity, Fairness) in real-time.

It features a robust **Offline-to-Online** pipeline, a **Generative User Simulator** for realistic training, and support for **Combinatorial Slate Recommendations**.

## Core Architectures

### 1. Hierarchical & Distributional RL
- **Weight Agent**: A Meta-Controller that dynamically adjusts the importance weights $w$ for multiple objectives based on the user's state. It uses **Distributional RL (Quantile Regression)** to estimate the distribution of future returns, optimizing for risk-aware metrics like **CVaR** (Conditional Value at Risk).
- **SAC Agent**: A Soft Actor-Critic agent that takes the state $s$ and the dynamic weights $w$, outputting the optimal action (or slate) to maximize the scalarized reward $r = w \cdot \vec{r}$.

### 2. Generative User Simulator
- **Latent State Dynamics**: Uses a **GRU** (Gated Recurrent Unit) to model the user's hidden mental state and intent evolution over time.
- **Stochastic Satisfaction**: Models user satisfaction as an **Ornstein-Uhlenbeck** process (diffusion), simulating realistic mood drifts and frustration.
- **Choice Model**: Simulates user interaction with slates using a **Multinomial Logit (MNL)** choice model.

### 3. Verification & Fairness
- **Offline Evaluation**: Includes **Doubly Robust (DR)** estimators to evaluate policies on static datasets (MovieLens) before online deployment.
- **Safe Exploration**: Incorporates **Behavior Cloning (BC)** and **Conservative Q-Learning (CQL)** to ensure safe online fine-tuning.
- **Fairness Constraints**: Explicitly optimizes for **Long-tail Exposure** (Inverse Propensity) and **Demographic Parity** (Persona-based fairness).

## Key Features
- **Combinatorial Slates**: Recommends lists of items (Slates) with submodular rewards (Diversity).
- **4-Objective MOMDP**: Optimizes for Engagement, Satisfaction, Diversity, and Fairness.
- **Granular Analysis**: Extensive visualization suite to inspect internal agent dynamics.

## Installation

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

## Usage

### 1. Training
To train the agent, run `train.py`. You can configure hyperparameters via CLI arguments.

**Basic Run (Online Only):**
```powershell
python train.py --max_episodes 100 --slate_size 3
```

**Hybrid Run (Offline Pre-training + Online Fine-tuning):**
Point to your MovieLens dataset folder (must contain `ratings.csv` and `movies.csv`).
```powershell
# Safe Offline-to-Online Transfer with CQL and Behavior Cloning
python train.py --dataset_path "C:/path/to/ml-latest/dataset" --cql_weight 1.0 --bc_weight 0.5 --max_episodes 200
```

### 2. Analysis & Visualization
DOM-RL includes a suite of tools to visualize the "Black Box" of the agent.

1. **Collect Granular Data**: Runs the trained agent to generate detailed logs.
   ```powershell
   python collect_granular_data.py
   ```
2. **Generate Gallery**: Creates ~40 plots in `logs/gallery/`.
   ```powershell
   python generate_gallery.py
   ```
3. **Plot Training Curves**:
   ```powershell
   python plot_training.py
   ```

### 3. Documentation
See [finds.md](finds.md) for a comprehensive, auto-generated report on the agent's behavior, including:
- Architecture Diagrams
- Mathematical Formulations (CQL, CVaR, Fairness)
- User Simulator Dynamics
- Benchmark Results

## Project Structure
- `domrl/`: Core package
    - `agent/`: `SACAgent` (CQL+BC) and `WeightAgent` (Distributional).
    - `env/`: `RealTimeRecEnv` and `GenerativeUserSimulator`.
    - `models/`: Neural Network architectures (`StateEncoder`, `UserDynamicsNet`).
    - `utils/`: `ReplayBuffer`, `SlateMapper`, `DataLoader`.
- `logs/`: Training logs and generated images.
- `train.py`: Main entry point.
- `evaluate.py`: Offline (DR) and Online evaluation scripts.

## License
MIT
