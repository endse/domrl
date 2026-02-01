# DOM-RL: Advanced Technical Deep Dive

**Dynamic Multi-Objective Reinforcement Learning with Generative User Simulation**

This document serves as the comprehensive technical reference for the DOM-RL framework. It details the system architecture, the new generative simulation engine, and the advanced reinforcement learning techniques used to solve the multi-objective slate recommendation problem. This document also includes a gallery of visualizations analyzing the agent's behavior.

---

# Part 1: System Architecture & Logic

## 1. System Ecosystem

The DOM-RL framework is composed of two hierarchical agents interacting with a complex user environment. The diagram below illustrates the flow of information and control.

### Visualization: The DOM-RL Control Loop
*This flowchart visualizes the cyclic interaction between the User, the Environment, and the two Agent tiers.*

```mermaid
graph TD

    %% =========================
    %% Generative Environment
    %% =========================
    subgraph Generative_Environment
        U(("User Hidden State")) -->|Signals| S[State Vector]
        S -->|Observation| WA
        S -->|Observation| SA
    end

    %% =========================
    %% Tier 1: Meta-Controller
    %% =========================
    subgraph Tier_1_Meta_Controller
        WA[Weight Agent] -->|"Distributional RL"| W[Objective Weights w]
        style WA fill:#f9f,stroke:#333,stroke-width:2px
    end

    %% =========================
    %% Tier 2: Policy Agent
    %% =========================
    subgraph Tier_2_Policy_Agent
        SA[SAC Agent] -->|"Conservative Q-Learning"| A[Action Index]
        W -->|Conditions| SA
        style SA fill:#bbf,stroke:#333,stroke-width:2px
    end

    %% =========================
    %% Slate Engine
    %% =========================
    subgraph Slate_Engine
        A -->|Maps to| SL{Slate Tuple}
        SL -->|Items| U
    end

    %% =========================
    %% Rewards and Updates
    %% =========================
    U -->|Feedback| R[Base Rewards]
    R -->|Scalarization| FR[Final Reward]
    FR -->|Update| WA
    FR -->|Update| SA

```

**Explanation:**
1.  **User**: Produces noisy signals (clicks, hovers) based on a hidden internal state.
2.  **Weight Agent (Meta-Controller)**: observes the state and determines the optimal balance of business objectives (e.g., "Prioritize Fairness vs. SAT").
3.  **SAC Agent (Policy)**: Takes both the state and the weights to generate a specific action.
4.  **Slate Engine**: Converts the abstract action index into a concrete list of 3 items (Slate) to show the user.

---

## 2. Generative User Simulator (Sim-V2)

We have moved from a heuristic-based simulator to a **Generative Model** that mimics realistic user psychology.

### Visualization: User State Machine
*This state diagram illustrates the internal lifecycle of the simulated user during a single interaction step.*

```mermaid
stateDiagram-v2

    [*] --> Latent_Update : Agent presents slate

    state "1. Latent Update - GRU" as Latent_Update
    note right of Latent_Update
        Update hidden mood state h_t
        using interaction history
    end note

    Latent_Update --> Choice_Model

    state "2. Choice Model - MNL" as Choice_Model
    note right of Choice_Model
        Evaluate items in slate
        Probabilistic selection via softmax
    end note

    Choice_Model --> Dynamics

    state "3. Satisfaction Dynamics - SDE" as Dynamics
    note right of Dynamics
        S_t drifts toward target satisfaction
        Applies diffusion noise
    end note

    Dynamics --> Signals

    state "4. Signal Generation" as Signals
    note right of Signals
        Generate click, hover, scroll
        Based on true intent match
    end note

    Signals --> [*] : Return observation


```

**Explanation:**
-   **GRU Core**: The user has a "memory" modeled by a Gated Recurrent Unit neural network.
-   **Choice Model**: The user doesn't just randomly click. They evaluate the slate using a Multinomial Logit model, picking the item that best matches their current intent.
-   **Diffusion**: Satisfaction isn't static. It drifts over time like a random walk (Ornstein-Uhlenbeck process), creating realistic "mood swings".

---

## 3. Offline-to-Online Transfer Pipeline

A key feature of DOM-RL is the ability to pre-train on static datasets (MovieLens) and transfer safely to the online simulator.

### Visualization: The Safety Pipeline
*This sequence diagram shows how a model graduates from offline data to online deployment.*

```mermaid
sequenceDiagram
    participant D as MovieLens Data
    participant A as SAC Agent
    participant C as Critic (CQL)
    participant S as Simulator

    Note over A, C: Phase 1: Offline Pre-Training
    
    loop Batch Training
        D->>A: Historical Transition (s, a, r, s')
        
        par Behavior Cloning
            A->>A: Minimize Divergence from Data Policy
        and Conservative Q-Learning
            C->>C: Penalize Q-values of Unseen Actions
        end
    end
    
    Note over A, S: Phase 2: Online Fine-Tuning
    
    loop Interaction
        S->>A: State s
        A->>S: Action a (Exploration)
        S->>A: Reward r (Real-time feedback)
        A->>A: Update Policy (Gradual shift)
    end
```

**Explanation:**
-   **Offline Phase**: We use **Behavior Cloning (BC)** to force the agent to mimic human curators initially. **CQL** ensures the critic doesn't overestimate rewards for actions it hasn't seen in the dataset.
-   **Online Phase**: Once deployed, the agent gradually explores. The constraints (BC/CQL) are relaxed, allowing better-than-human performance.

---

## 4. Fairness and Slate Logic

The rewards and actions have been extended to support ethical AI and complex recommendations.

### Visualization: Reward Composition (Pie Chart)
*This chart breaks down the components of the 4D Reward Vector used in the Multi-Objective MDP.*

```mermaid
pie title Reward Vector Components
    "Engagement (Click/View)" : 40
    "Satisfaction (User Happiness)" : 30
    "Diversity (Slate Submodularity)" : 15
    "Fairness (Exposure/Demographics)" : 15
```

**Explanation:**
-   **Engagement**: Standard interaction metrics (CTR).
-   **Satisfaction**: Long-term user retention proxy.
-   **Diversity**: A submodular penalty applied if the items in the slate are too similar (e.g., 3 Action movies).
-   **Fairness**:
    -   *Exposure*: Boosts reward for items that appear in the "Long Tail" (rarely visited).
    -   *Demographic*: Boosts reward for satisfying underserved persona groups (e.g., "Critics").

---

# Part 2: Comprehensive Model Analysis (Gallery)

The following sections analyze the agent's behavior using granular visualization data.

## 1. Chapter 1: The User Model (State Analysis)
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

## 2. Chapter 2: The Business Brain (Dynamic Weights)
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

## 3. Chapter 3: Decision Making (Actions)
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

---

## 4. Chapter 4: Correlations & Dynamics
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

## 5. Chapter 5: Episode Case Studies
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

## 6. Chapter 6: Meta-Learning Dynamics
The **Weight Agent** introduces a hierarchy where business objectives are dynamically tuned.

### Training Progress
![Training Summary](logs/gallery/training_summary.png)
*   **Panel 1 (Rewards)**: Shows the agent reliably maintaining satisfaction (Green line) while maximizing scalar reward.
*   **Panel 2 (Weights)**: The most critical insight. The Weight Agent does NOT converge to static weights. Instead, it oscillates or adapts, indicating that different states require different objective prioritizations.
*   **Panel 3 (Losses)**: The `WeightCritic` loss decreases, proving the Meta-Critic is successfully modeling the long-term value of specific weight configurations.

---

## 7. Chapter 7: Benchmarks & Personas
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

## 8. Chapter 8: Feature Analysis - HITL & Context
**New Features (v2.0)**: We introduced real-time Human-in-the-Loop feedback and Context Awareness.

### Context Awareness: User Mood
How does the simulator adapt to "Emotional Context"?
![Mood Sad](logs/gallery/context_mood_sad.png)
*   **Validation**: When the user context is set to **"Sad"**, the preference distribution shifts significantly towards **Comedy** (Cheer up) and **Drama** (wallow). The agent must learn to detect this shift or rely on the explicit context signal.

### Temporal Dynamics: Time of Day
![Time Night](logs/gallery/context_time_night.png)
*   **Validation**: At **23:00 (Night)**, the preference for **Horror** and **Thriller** spikes, while "Daytime" genres like Documentary drop. This proves the Time-of-Day control effectively biases the simulation.

### Human-in-the-Loop Adaptation
Does the "Like" button actually work?
![HITL Boost](logs/gallery/hitl_enthusiasm_boost.png)
*   **Validation**: A manual **"Like"** interaction (at Step 5) causes an immediate, stepwise jump in **User Enthusiasm**. This triggers the system to transition from explore/safety mode to exploitation (high reward state).

### Persona DNA Analysis (Radar Chart)
We analyzed the theoretical "Taste Profiles" of our 4 simulated personas to ensure distinct behavior.
![Radar Persona](logs/gallery/radar_personas.png)
*   **Insight**: The **Standard** user (Blue) has a balanced, circular profile. The **Critic** (Red) shows sharp spikes for "Serious" genres (Drama, Doc, Noir) while rejecting "Popcorn" movies. The **Binger** (Green) consumes almost everything but leans towards entertainment.

### The "Mood Matrix" (Heatmap)
How does internal state bias external action?
![Mood Matrix](logs/gallery/heatmap_mood_bias.png)
*   **Insight**: This heatmap reveals the hidden **Logit Bias** applied by the simulator.
    *   **Red Zones**: High positive bias (e.g., Happy -> Comedy).
    *   **Blue Zones**: Negative bias (e.g., Tired -> Thriller).
    *   This matrix proves the environment is not static; it "breathes" based on user context.

---
*Generated for DOM-RL Project. Updated 2026-02-01.*
