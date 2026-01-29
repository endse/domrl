# DOM-RL

Dynamic Multi-Objective Deep Reinforcement Learning for Real-Time Recommendation.

## Overview
DOM-RL is a reinforcement learning framework designed to optimize recommendation systems by balancing multiple conflicting objectives:
1.  **Engagement**: Maximizing user clicks and interactions.
2.  **Satisfaction**: Inferring user satisfaction from micro-behaviors (hover durations, scroll velocity).
3.  **Churn Prevention**: Minimizing the risk of user session termination.

## Architecture
- **Environment**: Custom Gymnasium environment simulating user browsing behavior.
- **Agent**: Soft Actor-Critic (SAC) with discrete action space (Discrete SAC).
- **Reward**: Multi-objective scalarization: $R = w_1 \cdot E + w_2 \cdot S - w_3 \cdot C$.

## Setup
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
Train the agent:
```bash
python train.py
```
This will run the simulation and save checkpoints to `logs/`.

## MovieLens Retraining
The framework supports offline training using the MovieLens dataset.
1. Ensure the dataset is located at `c:\Users\cy569\Downloads\ml-latest\dataset` (or modify `train.py`).
2. Run `python train.py`. 
   - The script will automatically detect the dataset, perform offline pre-training, and then switch to online fine-tuning (Hybrid Mode).

## Evaluation
To verify performance and generate analysis plots:
```bash
python evaluate.py         # Runs scenarios
python collect_granular_data.py # Collects detailed trace
python generate_gallery.py # Creating visualizations for finds.md
```
See `finds.md` for the comprehensive analysis.
