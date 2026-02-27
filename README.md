# DOM-RL: Dynamic Multi-Objective Deep Reinforcement Learning for Real-Time Recommendation

A framework for real-time adaptive recommendations that moves beyond static, offline-trained models. DOM-RL tracks micro-behavioral signals (scroll velocity, hover duration, skip-rate gradients) and uses a **Hybrid SAC-NSGA-II** architecture to balance conflicting objectives: engagement, satisfaction, diversity, fairness, and churn mitigation.

## Architecture

```
User Session
    │
    ▼
┌──────────────────────────┐
│  Micro-Behavioral Tracker │  ← Scroll Velocity, Hover-Dwell, Skip Gradients
│  (UserDynamicsNet + GRU)  │
└──────────┬───────────────┘
           │ State (6-dim micro + history + persona)
           ▼
┌──────────────────────────┐     ┌──────────────────────┐
│     SAC Agent (Actor)     │◄────│   Weight Agent        │
│  Gaussian Policy + tanh   │     │  (Hybrid SAC-NSGA-II) │
│  Twin Critics (Q1, Q2)    │     │  5-Objective Weights   │
└──────────┬───────────────┘     └──────────┬───────────┘
           │ Action Embedding (16-dim)       │ Pareto-optimal weights
           ▼                                 │
┌──────────────────────────┐                 │
│   Recommendation Env      │◄───────────────┘
│  Slate → Simulator → Reward│
│  5 Objectives:             │
│   1. Engagement (CTR)      │
│   2. Satisfaction          │
│   3. Diversity             │
│   4. Fairness              │
│   5. Churn Mitigation      │
└──────────────────────────┘
```

## Key Features

- **Granular Feature Engineering (GFE)**: Extracts intent from scroll velocity, hover-dwell ratios, and skip-rate temporal gradients
- **Soft Actor-Critic (SAC)**: Maximum entropy RL with twin critics, dynamic temperature α, and tanh squashing correction
- **NSGA-II Optimizer**: Non-dominated sorting, crowding distance, SBX crossover for Pareto-optimal weight discovery
- **Hybrid SAC-NSGA-II**: NSGA-II pre-optimizes the weight space; SAC fine-tunes via gradient descent
- **Cold Start Inference**: Infers user persona from navigation patterns within first 5 interactions
- **5-Objective Optimization**: Engagement, Satisfaction, Diversity, Fairness, Churn Mitigation

## Project Structure

```
domrl/
├── agent/
│   ├── sac.py              # SAC agent (entropy-regularized policy)
│   ├── weight_agent.py     # Hybrid SAC-NSGA-II weight optimization
│   ├── nsga2.py            # NSGA-II (non-dominated sorting, crowding distance)
│   └── baselines.py        # Random/Static baseline agents
├── env/
│   ├── rec_env.py          # Gymnasium environment (5-objective MOMDP)
│   └── user_simulator.py   # Generative user simulator with micro-behavioral signals
├── models/
│   └── networks.py         # Actor, Critic, StateEncoder (LayerNorm, 6-dim micro)
├── utils/
│   ├── replay_buffer.py    # Experience replay (6-dim micro, 5-dim weights)
│   ├── data_loader.py      # Netflix/MovieLens data loading
│   ├── movie_db.py         # Movie database with ANN search
│   └── slate_utils.py      # Slate combinatorial mapper
└── config.py               # Configuration (NSGA-II params, thresholds)
```

## Quick Start

### Verify Installation
```bash
python verify_v3.py
```

### Train
```bash
python train.py --max_episodes 2000 --slate_size 3
```

### Key Arguments
| Argument | Default | Description |
|---|---|---|
| `--max_episodes` | 2000 | Training episodes |
| `--nsga2_pop_size` | 50 | NSGA-II population size |
| `--nsga2_generations` | 20 | Generations per evolution |
| `--nsga2_evolve_interval` | 10 | Episodes between evolutions |
| `--slate_size` | 3 | Recommendation slate size |

## Paper Reference

> **Dynamic Multi-Objective Deep Reinforcement Learning for Real-Time Recommendation (DOM-RL)**
>
> Keywords: SAC, NSGA-II, MORL, Micro-behavioral Tracking, Netflix 1M Dataset

## Requirements

- Python 3.10+
- PyTorch
- Gymnasium
- pandas, numpy
- tensorboard
- python-dotenv
