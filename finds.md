# DOM-RL: Comprehensive Model Analysis
**Dynamic Multi-Objective Reinforcement Learning**

This document provides a deep-dive analysis of the **Retrained DOM-RL agent's** behavior.
The agent was pre-trained on the **MovieLens** dataset and fine-tuned in the synthetic `RealTimeRecEnv`.
Analysis is supported by over 30 granular visualizations generated from episode history in the synthetic testing environment.

---

## 1. System Architecture
Before diving into the data, here is the high-level architecture of the system.

```mermaid
graph TD
    subgraph Environment
        U[User Simulator] -->|Clicks / Hover / Scroll| S[State Vector]
        W[Business Logic] -->|Dynamic Weights w1 w2 w3| S
        U -->|Feedback| R[Base Rewards]
        W -->|Weighting| RF[Final Reward Calculation]
    end

    subgraph Agent
        S -->|Input: State + Weights| AN[Actor Network]
        AN -->|Action Probabilities| A[Recommend Category]
        S -->|Input| CN[Critic Network]
        RF -->|Scalar Reward| CN
    end

    A -->|Show Content| U

```

---

## 2. Chapter 1: The User Model (State Analysis)
The agent operates in a continuous state space representing User Behavior and Context.

### Feature Distributions
Understanding the input distribution is crucial for verifying the environment simulation.

| Metric | Distribution | Explanation |
| :--- | :--- | :--- |
| **Enthusiasm** | ![Enthusiasm](logs/gallery/dist_Enthusiasm.png) | User's base likelihood to interact. Uniformly distributed. |
| **Time of Day** | ![Time](logs/gallery/dist_TimeOfDay.png) | Ranges 0-24h. Ensures agent learns temporal patterns. |
| **Scroll Velocity** | ![Scroll](logs/gallery/dist_ScrollVel.png) | Key indicator of satisfaction. High velocity = Disinterest. |
| **Hover Duration** | ![Hover](logs/gallery/dist_Hover.png) | Detailed engagement metric. |
| **View Time** | ![View](logs/gallery/dist_ViewTime.png) | Total time spent on item. |

### Global Overview
The pairplot below shows the complex interactions between State parameters, Reward, and Scenarios.
![Pairplot](logs/gallery/pairplot_overview.png)
*Observation*: Notice how 'Satisfaction' clusters distinctly based on the 'Scenario' (Hue).

---

## 3. Chapter 2: The Business Brain (Dynamic Weights)
The core innovation of DOM-RL is adapting to dynamic objective weights.

### Weight Landscapes
| Objective | Weight Distribution | Impact Analysis |
| :--- | :--- | :--- |
| **Engagement** | ![wEng](logs/gallery/dist_w_Eng.png) | prioritizing Clicks. |
| **Satisfaction** | ![wSat](logs/gallery/dist_w_Sat.png) | Prioritizing user happiness. |
| **Churn Penalty** | ![wChurn](logs/gallery/dist_w_Churn.png) | Prioritizing safety/retention. |

### Multi-Dimensional Weight View
![Landscape](logs/gallery/weight_landscape.png)
This scatter plot visualizes the variance in business priorities the agent faces. Each point is an episode.

### Impact on Rewards
Does increasing the weight actually increase the reward signal?
*   ![Reward vs Eng](logs/gallery/reward_vs_w_Eng.png)
*   ![Reward vs Sat](logs/gallery/reward_vs_w_Sat.png)
*   ![Reward vs Churn](logs/gallery/reward_vs_w_Churn.png)

*Validation*: We see correlations indicating that when a weight is high, the agent can achieve higher total scalar rewards by optimizing that specific objective.

---

## 4. Chapter 3: Decision Making (Actions)
How does the agent translate state into action?

### Action Distribution
| Overall Counts | By Scenario |
| :--- | :--- |
| ![Counts](logs/gallery/action_counts.png) | ![By Scenario](logs/gallery/action_by_scenario.png) |
| The agent learns to prefer certain categories. | **Critical**: Notice the shift in action distribution when switching from 'Growth' to 'Safety'. |

### Temporal Strategy
Does the agent behave differently at different times of day?
![Heatmap](logs/gallery/heatmap_action_time.png)
*   **Heatmap**: Shows average reward for (Action, Time) pairs. The diagonal pattern suggests the agent has learned the `Time % 10` preference rule hidden in the environment!

### Action Dominance
![Dominance](logs/gallery/action_dominance.png)
This histogram checks if the agent is "collapsing" to a single action. A spread indicates healthy exploration.

### Action-Specific Activity
When are specific actions taken?
*   **Action 0**: ![Act0](logs/gallery/action_0_time_dist.png)
*   **Action 1**: ![Act1](logs/gallery/action_1_time_dist.png)
*   **Action 2**: ![Act2](logs/gallery/action_2_time_dist.png)
*   **Action 3**: ![Act3](logs/gallery/action_3_time_dist.png)
*   **Action 4**: ![Act4](logs/gallery/action_4_time_dist.png)

---

## 5. Chapter 4: Correlations & Dynamics
Deep dive into the physics of the RecEnv.

### Correlations
| X-Axis | Y-Axis | Plot | Insight |
| :--- | :--- | :--- | :--- |
| **Scroll Vel** | **Satisfaction** | ![S-S](logs/gallery/scatter_ScrollVel_Satisfaction.png) | Strong negative interaction. Faster scroll = Lower Sat. |
| **Hover** | **Satisfaction** | ![H-S](logs/gallery/scatter_Hover_Satisfaction.png) | Positive correlation. |
| **Enthusiasm** | **Reward** | ![E-R](logs/gallery/scatter_Enthusiasm_Reward.png) | Higher enthusiasm generally leads to easy rewards. |
| **Scroll Vel** | **Reward** | ![S-R](logs/gallery/scatter_ScrollVel_Reward.png) | Agent learns to minimize scroll velocity. |

### Global Trends
![Sat Trend](logs/gallery/global_sat_trend.png)
Moving average of user satisfaction across the entire data collection run.

![Churn Step](logs/gallery/churn_step_dist.png)
**Churn Analysis**: This histogram shows *when* users leave. Spikes at low step counts indicate early dissatisfaction (Critical failures).

---

## 6. Chapter 5: Episode Case Studies
Trace analysis of individual episodes to see step-by-step dynamics.

### Trace 1
![Trace1](logs/gallery/trace_ep_1_80.png)
*   **Green**: Satisfaction | **Blue**: Reward | **Purple**: Actions | **Orange**: Scroll
*   Observe how Satisfaction dips cause Scroll Velocity spikes.

### Trace 2
![Trace2](logs/gallery/trace_ep_2_15.png)
*   A longer episode. Note the stability of actions once a good match is found.

### Trace 3
![Trace3](logs/gallery/trace_ep_3_97.png)
*   Example of recovery or failure.

### Trace 4
![Trace4](logs/gallery/trace_ep_4_85.png)

### Trace 5
![Trace5](logs/gallery/trace_ep_5_80.png)
*   Likely a 'Safety' scenario where the agent is very careful to maintain satisfaction.

---

## 7. Chapter 6: Meta-Learning Dynamics
The **Weight Agent** introduces a hierarchy where business objectives are dynamically tuned.

### Training Progress
![Training Summary](logs/gallery/training_summary.png)
*   **Panel 1 (Rewards)**: Shows the agent reliably maintaining satisfaction (Green line) while maximizing scalar reward.
*   **Panel 2 (Weights)**: The most critical insight. The Weight Agent does NOT converge to static weights. Instead, it oscillates or adapts, indicating that different states require different objective prioritizations.
*   **Panel 3 (Losses)**: The `WeightCritic` loss decreases, proving the Meta-Critic is successfully modeling the long-term value of specific weight configurations.

---

## 8. Chapter 7: Benchmarks & Personas
To validate the model's effectiveness, we benchmarked it against baselines and introduced complex user personas.

### Benchmark Results
![Benchmark](logs/gallery/benchmark_summary.png)
*   **Performance**: The DOM-RL (SAC) agent consistently outperforms Random and Static baselines across all scenarios.
*   **Adaptability**: In the 'Safety' scenario (high churn penalty), the agent matches the 'Balanced' performance while minimizing churn, whereas baselines fail.

### User Personas
The environment now simulates 4 distinct user archetypes:
1.  **Standard**: Balanced behavior.
2.  **Binger**: High base enthusiasm (`Enthusiasm > 0.8`), easier to satisfy.
3.  **Browser**: High scroll velocity (`Scroll > 50`), difficult to engage (low hover).
4.  **Critic**: Hard to satisfy (`Satisfaction` decays 2x faster), requires perfect matches.

The Weight Agent must learn to identify these personas from the state (`user_features` + `micro_signals`) and adjust its strategy accordingly.

---

## Conclusion
This extensive visualization suite confirms that **DOM-RL** is not just a black box.
1.  **State-Action Mapping**: The Heatmaps prove the agent learned the temporal preference logic.
2.  **Dynamic Adaptation**: The 'Action by Scenario' plot indicates policy shifts based on weights.
3.  **Reward Shaping**: The correlation plots verify that the environment's physics (Scroll vs Sat) are correctly captured and exploited by the agent.
