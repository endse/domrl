# DOM-RL

**Dynamic Multi-Objective Deep Reinforcement Learning for Real-Time Recommendation**

DOM-RL is an advanced recommendation framework that uses Hierarchical Reinforcement Learning to dynamically balance conflicting business objectives (Engagement, Satisfaction, Diversity) in real-time.

## Key Features
- **Dynamic Weighting**: A "Weight Agent" (Meta-Controller) learns to adjust the importance of different objectives based on the user's current state.
- **Meta-Learning**: Includes a **Meta-Critic** that predicts long-term user satisfaction given a specific weighting strategy.
- **Hybrid Training**: Supports offline pre-training on MovieLens followed by online fine-tuning in a simulated environment.
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
python train.py --max_episodes 100
```

**Hybrid Run (Offline Pre-training + Online Fine-tuning):**
Point to your MovieLens dataset folder (must contain `ratings.csv` and `movies.csv`).
```powershell
python train.py --dataset_path "C:/path/to/ml-latest/dataset" --max_episodes 200
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
- State-Action Heatmaps
- Weight Adaptation Landscapes
- Episode Traces

## Project Structure
- `domrl/`: Core package
    - `agent/`: SAC and WeightAgent implementations.
    - `env/`: `RealTimeRecEnv` (Gymnasium environment).
    - `models/`: Neural Network architectures (`StateEncoder`, `MetaCritic`).
- `logs/`: Training logs and generated images.
- `train.py`: Main entry point.

## License
MIT
