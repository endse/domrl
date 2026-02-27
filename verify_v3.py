"""
DOM-RL v3.0 Verification Script

Validates all paper-aligned components:
1. Expanded micro-behavioral signals (6-dim)
2. 5-objective reward vector
3. NSGA-II non-dominated sorting
4. Cold start persona inference
5. Hybrid SAC-NSGA-II training loop
6. End-to-end training for 3 episodes
"""

import numpy as np
import torch
import os
from domrl.env.rec_env import RealTimeRecEnv
from domrl.agent.sac import SACAgent
from domrl.agent.weight_agent import WeightAgent
from domrl.utils.replay_buffer import ReplayBuffer
from domrl.agent.nsga2 import (
    non_dominated_sort, crowding_distance, 
    NSGA2Optimizer, dominates
)


def verify_nsga2():
    """Test NSGA-II non-dominated sorting on a known example."""
    print("=" * 60)
    print("TEST 1: NSGA-II Non-Dominated Sorting")
    print("=" * 60)
    
    # Known test case: 4 points in 2D objective space
    objectives = np.array([
        [3.0, 1.0],  # Pareto front
        [1.0, 3.0],  # Pareto front
        [2.0, 2.0],  # Pareto front
        [1.0, 1.0],  # Dominated
    ])
    
    fronts = non_dominated_sort(objectives)
    print(f"  Fronts: {fronts}")
    assert len(fronts) >= 2, f"Expected at least 2 fronts, got {len(fronts)}"
    assert 3 not in fronts[0], "Point (1,1) should be dominated"
    assert set(fronts[0]) == {0, 1, 2}, f"Front 0 should be {{0,1,2}}, got {set(fronts[0])}"
    print("  ✓ Non-dominated sorting correct")
    
    # Test crowding distance
    distances = crowding_distance(objectives, fronts[0])
    print(f"  Crowding distances: {distances}")
    assert len(distances) == 3
    print("  ✓ Crowding distance computed")
    
    # Test dominates
    assert dominates(np.array([3.0, 3.0]), np.array([1.0, 1.0]))
    assert not dominates(np.array([3.0, 1.0]), np.array([1.0, 3.0]))
    print("  ✓ Domination check correct")
    
    # Test NSGA2Optimizer
    optimizer = NSGA2Optimizer(num_objectives=5, pop_size=20, num_generations=5)
    assert optimizer.population.shape == (20, 5)
    
    # Set some dummy objectives and evolve
    dummy_obj = np.random.rand(20, 5)
    optimizer.set_objectives(dummy_obj)
    pareto_weights = optimizer.evolve()
    
    assert len(pareto_weights) > 0, "NSGA-II should produce Pareto weights"
    best = optimizer.get_best_weights("balanced")
    assert best.shape == (5,), f"Best weights should be (5,), got {best.shape}"
    print(f"  ✓ NSGA-II evolution produced {len(pareto_weights)} Pareto-optimal solutions")
    print(f"  Best balanced weights: {best}")
    
    hv = optimizer.get_hypervolume()
    print(f"  Hypervolume: {hv:.4f}")
    print("  ✓ NSGA-II module fully functional")
    print()


def verify_environment():
    """Test expanded environment: 6 micro-signals, 5 objectives, cold start."""
    print("=" * 60)
    print("TEST 2: Environment (Expanded Signals & Objectives)")
    print("=" * 60)
    
    env = RealTimeRecEnv()
    obs, _ = env.reset()
    
    # Check observation dimensions
    assert obs['micro_signals'].shape == (6,), f"Expected micro_signals (6,), got {obs['micro_signals'].shape}"
    assert obs['weights'].shape == (5,), f"Expected weights (5,), got {obs['weights'].shape}"
    print(f"  ✓ Observation keys: {obs.keys()}")
    print(f"  ✓ micro_signals shape: {obs['micro_signals'].shape} (6-dim)")
    print(f"  ✓ weights shape: {obs['weights'].shape} (5-dim)")
    print(f"  ✓ Action space: {env.action_space}")
    
    # Test step with random action
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)
    
    reward_vec = info['reward_vector']
    assert reward_vec.shape == (5,), f"Expected reward_vector (5,), got {reward_vec.shape}"
    print(f"  ✓ Reward vector: {reward_vec} (5 objectives)")
    print(f"  ✓ Churn probability: {info.get('churn_probability', 'N/A')}")
    print(f"  ✓ Cold start: {info.get('cold_start', 'N/A')}")
    
    # Check enriched signals
    signals = info['signals']
    assert 'scroll_velocity' in signals, "Missing scroll_velocity signal"
    assert 'hover_dwell_ratio' in signals, "Missing hover_dwell_ratio signal"
    assert 'skip_gradient' in signals, "Missing skip_gradient signal"
    print(f"  ✓ Enriched signals present: scroll_velocity={signals['scroll_velocity']:.2f}, "
          f"hover_dwell_ratio={signals['hover_dwell_ratio']:.2f}, "
          f"skip_gradient={signals['skip_gradient']:.2f}")
    print()


def verify_agents():
    """Test SAC and WeightAgent with expanded dimensions."""
    print("=" * 60)
    print("TEST 3: Agent Initialization & Action Selection")
    print("=" * 60)
    
    env = RealTimeRecEnv()
    obs, _ = env.reset()
    
    action_dim = env.action_space.shape[0]
    state_dim = 0
    
    sac_agent = SACAgent(state_dim, action_dim, num_items=env.num_categories)
    weight_agent = WeightAgent(
        action_dim=action_dim, num_items=env.num_categories,
        num_objectives=env.num_objectives
    )
    
    # Test action selection
    action = sac_agent.select_action(obs)
    assert action.shape == (action_dim,), f"Expected action ({action_dim},), got {action.shape}"
    print(f"  ✓ SAC action shape: {action.shape}")
    
    weights = weight_agent.select_weights(obs)
    assert weights.shape == (5,), f"Expected weights (5,), got {weights.shape}"
    print(f"  ✓ Weight agent output: {weights}")
    print()


def verify_replay_buffer():
    """Test replay buffer with expanded dimensions."""
    print("=" * 60)
    print("TEST 4: Replay Buffer (Expanded Dimensions)")
    print("=" * 60)
    
    env = RealTimeRecEnv()
    obs, _ = env.reset()
    
    action_dim = env.action_space.shape[0]
    buf = ReplayBuffer(0, action_dim, max_size=100, micro_dim=6, num_objectives=5)
    
    # Add some transitions
    for i in range(5):
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        buf.add(obs, action, next_obs, reward, done, meta_reward=float(reward))
        obs = next_obs
        if done: obs, _ = env.reset()
    
    assert buf.size == 5
    print(f"  ✓ Buffer size: {buf.size}")
    
    # Sample
    state, action, next_state, reward, not_done, meta_reward = buf.sample(2)
    assert state['micro_signals'].shape == (2, 6), f"Expected (2,6), got {state['micro_signals'].shape}"
    assert state['weights'].shape == (2, 5), f"Expected (2,5), got {state['weights'].shape}"
    print(f"  ✓ Sampled micro_signals: {state['micro_signals'].shape}")
    print(f"  ✓ Sampled weights: {state['weights'].shape}")
    print()


def verify_agent_updates():
    """Test gradient updates for both agents."""
    print("=" * 60)
    print("TEST 5: Agent Gradient Updates")
    print("=" * 60)
    
    env = RealTimeRecEnv()
    obs, _ = env.reset()
    
    action_dim = env.action_space.shape[0]
    sac_agent = SACAgent(0, action_dim, num_items=env.num_categories)
    weight_agent = WeightAgent(
        action_dim=action_dim, num_items=env.num_categories,
        num_objectives=env.num_objectives
    )
    buf = ReplayBuffer(0, action_dim, max_size=100, micro_dim=6, num_objectives=5)
    
    # Fill buffer
    for i in range(10):
        weights = weight_agent.select_weights(obs)
        env.weights = weights
        obs['weights'] = weights
        
        action = sac_agent.select_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Record for NSGA-II
        weight_agent.record_reward_vector(info['reward_vector'])
        
        buf.add(obs, action, next_obs, reward, done, meta_reward=float(reward))
        obs = next_obs
        if done: obs, _ = env.reset()
    
    # SAC Update
    critic_loss, actor_loss, alpha, q_vals, entropy = sac_agent.update(buf, batch_size=4)
    print(f"  ✓ SAC Update: critic_loss={critic_loss:.4f}, actor_loss={actor_loss:.4f}, "
          f"alpha={alpha:.4f}, entropy={entropy:.4f}")
    
    # Weight Agent Update
    w_critic_loss, w_actor_loss = weight_agent.update(buf, batch_size=4)
    print(f"  ✓ Weight Update: critic_loss={w_critic_loss:.4f}, actor_loss={w_actor_loss:.4f}")
    print()


def verify_cold_start():
    """Test cold start persona inference."""
    print("=" * 60)
    print("TEST 6: Cold Start Persona Inference")
    print("=" * 60)
    
    from domrl.env.user_simulator import GenerativeUserSimulator
    
    sim = GenerativeUserSimulator()
    sim.reset_state(persona_id=0)
    
    # Simulate a few steps to collect cold start data
    for i in range(6):
        sat, signals = sim.step([0, 1, 2])
    
    inferred_persona = sim.infer_cold_start_persona()
    print(f"  ✓ Inferred persona: {inferred_persona} (from scroll/hover patterns)")
    assert 0 <= inferred_persona <= 3, f"Invalid persona: {inferred_persona}"
    print(f"  ✓ Cold start scroll pattern: {sim.cold_start_scroll_pattern}")
    print(f"  ✓ Cold start hover pattern: {sim.cold_start_hover_pattern}")
    print()


def verify_hybrid_training():
    """Test 3-episode hybrid SAC-NSGA-II training."""
    print("=" * 60)
    print("TEST 7: Hybrid SAC-NSGA-II Training (3 Episodes)")
    print("=" * 60)
    
    env = RealTimeRecEnv()
    action_dim = env.action_space.shape[0]
    
    sac_agent = SACAgent(0, action_dim, num_items=env.num_categories)
    weight_agent = WeightAgent(
        action_dim=action_dim, num_items=env.num_categories,
        num_objectives=env.num_objectives,
        nsga2_pop_size=10, nsga2_generations=3
    )
    buf = ReplayBuffer(0, action_dim, max_size=1000, micro_dim=6, num_objectives=5)
    
    total_steps = 0
    
    for ep in range(3):
        obs, _ = env.reset()
        ep_reward = 0
        
        for step in range(20):  # Short episodes for verification
            weights = weight_agent.select_weights(obs)
            env.weights = weights
            obs['weights'] = weights
            
            action = sac_agent.select_action(obs) if total_steps > 10 else env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            
            weight_agent.record_reward_vector(info['reward_vector'])
            buf.add(obs, action, next_obs, reward, done, meta_reward=float(reward))
            
            obs = next_obs
            ep_reward += reward
            total_steps += 1
            
            if total_steps > 15:
                sac_agent.update(buf, batch_size=4)
                weight_agent.update(buf, batch_size=4)
            
            if done: break
        
        print(f"  Episode {ep+1}: Reward={ep_reward:.2f}, Sat={env.user_satisfaction:.2f}")
    
    # Test NSGA-II evolution
    nsga2_result = weight_agent.evolve_nsga2()
    if nsga2_result is not None:
        print(f"  ✓ NSGA-II produced weights: {nsga2_result}")
    else:
        print(f"  ✓ NSGA-II needs more data (expected for 3 short episodes)")
    
    hv = weight_agent.get_nsga2_hypervolume()
    print(f"  ✓ Hypervolume: {hv:.4f}")
    print(f"  ✓ Hybrid training completed successfully!")
    print()


def main():
    print("=" * 60)
    print("DOM-RL v3.0 Verification — Paper Alignment Check")
    print("=" * 60)
    print()
    
    try:
        verify_nsga2()
        verify_environment()
        verify_agents()
        verify_replay_buffer()
        verify_agent_updates()
        verify_cold_start()
        verify_hybrid_training()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
