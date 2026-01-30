"""
Comprehensive Test Suite for Day 101: Q-Learning Algorithm
Tests Q-Learning agent, Grid World environment, and training dynamics
"""

import pytest
import numpy as np
from lesson_code import GridWorld, QLearningAgent, train_agent


class TestGridWorld:
    """Test Grid World environment mechanics"""
    
    def test_initialization(self):
        """Test environment initializes correctly"""
        env = GridWorld(size=5)
        assert env.size == 5
        assert env.start == (0, 0)
        assert env.goal == (4, 4)
        assert len(env.obstacles) == 3
        assert env.n_actions == 4
    
    def test_reset(self):
        """Test environment resets to start position"""
        env = GridWorld(size=5)
        env.state = (3, 3)
        state = env.reset()
        assert state == (0, 0)
        assert env.state == (0, 0)
    
    def test_valid_action(self):
        """Test taking valid action updates state"""
        env = GridWorld(size=5)
        env.reset()
        next_state, reward, done, _ = env.step(1)  # Move down
        assert next_state == (1, 0)
        assert reward == -1  # Step penalty
        assert done is False
    
    def test_boundary_collision(self):
        """Test hitting boundary keeps agent in place"""
        env = GridWorld(size=5)
        env.reset()
        next_state, reward, done, _ = env.step(0)  # Move up from (0,0)
        assert next_state == (0, 0)  # Stay in place
        assert reward == -1
        assert done is False
    
    def test_obstacle_collision(self):
        """Test hitting obstacle terminates episode"""
        env = GridWorld(size=5, obstacles=[(1, 0)])
        env.reset()
        next_state, reward, done, _ = env.step(1)  # Move down into obstacle
        assert reward == -100
        assert done is True
    
    def test_goal_reached(self):
        """Test reaching goal gives large reward"""
        env = GridWorld(size=5)
        env.state = (4, 3)  # One step from goal
        next_state, reward, done, _ = env.step(3)  # Move right to goal
        assert next_state == (4, 4)
        assert reward == 100
        assert done is True
    
    def test_markov_property(self):
        """Test state transitions are deterministic (Markov)"""
        env = GridWorld(size=5)
        
        # Take same sequence of actions twice
        env.reset()
        states1 = [env.state]
        for action in [1, 3, 1, 3]:
            next_state, _, _, _ = env.step(action)
            states1.append(next_state)
        
        env.reset()
        states2 = [env.state]
        for action in [1, 3, 1, 3]:
            next_state, _, _, _ = env.step(action)
            states2.append(next_state)
        
        assert states1 == states2  # Should be identical
    
    def test_invalid_action(self):
        """Test invalid action raises error"""
        env = GridWorld(size=5)
        with pytest.raises(ValueError):
            env.step(5)  # Invalid action


class TestQLearningAgent:
    """Test Q-Learning agent functionality"""
    
    def test_initialization(self):
        """Test agent initializes with correct parameters"""
        agent = QLearningAgent(n_actions=4, learning_rate=0.1, 
                              discount_factor=0.9, epsilon=0.2)
        assert agent.n_actions == 4
        assert agent.alpha == 0.1
        assert agent.gamma == 0.9
        assert agent.epsilon == 0.2
        assert len(agent.q_table) == 0  # Empty at start
    
    def test_q_value_retrieval(self):
        """Test Q-value retrieval for unseen states"""
        agent = QLearningAgent(n_actions=4)
        state = (0, 0)
        action = 0
        
        # Unseen state-action should return 0.0
        q_value = agent.get_q_value(state, action)
        assert q_value == 0.0
    
    def test_q_value_update(self):
        """Test Q-value updates according to Bellman equation"""
        agent = QLearningAgent(n_actions=4, learning_rate=0.1, discount_factor=0.9)
        
        state = (0, 0)
        action = 1
        reward = 10
        next_state = (1, 0)
        done = False
        
        # Set Q-value for next state
        agent.q_table[(next_state, 0)] = 5.0
        agent.q_table[(next_state, 1)] = 8.0  # Max Q-value
        
        # Initial Q-value is 0
        initial_q = agent.get_q_value(state, action)
        
        # Update
        td_error = agent.update(state, action, reward, next_state, done)
        
        # Expected: Q(s,a) ← 0 + 0.1 * [10 + 0.9 * 8 - 0] = 1.72
        expected_q = 0.1 * (reward + 0.9 * 8.0)
        actual_q = agent.get_q_value(state, action)
        
        assert abs(actual_q - expected_q) < 1e-6
        assert td_error > 0
    
    def test_terminal_state_update(self):
        """Test Q-value update for terminal state (no future rewards)"""
        agent = QLearningAgent(n_actions=4, learning_rate=0.1, discount_factor=0.9)
        
        state = (4, 3)
        action = 3
        reward = 100
        next_state = (4, 4)
        done = True
        
        agent.update(state, action, reward, next_state, done)
        
        # Terminal state: target = reward only (no future)
        # Q(s,a) ← 0 + 0.1 * [100 - 0] = 10.0
        expected_q = 0.1 * reward
        actual_q = agent.get_q_value(state, action)
        
        assert abs(actual_q - expected_q) < 1e-6
    
    def test_epsilon_greedy_exploration(self):
        """Test epsilon-greedy action selection balances exploration/exploitation"""
        agent = QLearningAgent(n_actions=4, epsilon=0.5)
        
        # Set Q-values to make action 2 clearly best
        state = (0, 0)
        agent.q_table[(state, 0)] = 1.0
        agent.q_table[(state, 1)] = 1.0
        agent.q_table[(state, 2)] = 10.0  # Best action
        agent.q_table[(state, 3)] = 1.0
        
        # Run many trials
        actions = [agent.select_action(state, training=True) for _ in range(1000)]
        
        # Should mostly pick action 2, but sometimes explore others
        action_counts = np.bincount(actions, minlength=4)
        assert action_counts[2] > 400  # Best action chosen frequently
        assert action_counts[0] > 50   # Other actions explored
        assert action_counts[1] > 50
        assert action_counts[3] > 50
    
    def test_greedy_exploitation(self):
        """Test greedy policy always picks best action"""
        agent = QLearningAgent(n_actions=4, epsilon=0.0)
        
        state = (0, 0)
        agent.q_table[(state, 0)] = 1.0
        agent.q_table[(state, 1)] = 5.0  # Best action
        agent.q_table[(state, 2)] = 2.0
        agent.q_table[(state, 3)] = 3.0
        
        # Should always pick action 1
        actions = [agent.select_action(state, training=False) for _ in range(100)]
        assert all(a == 1 for a in actions)
    
    def test_epsilon_decay(self):
        """Test epsilon decreases over time"""
        agent = QLearningAgent(n_actions=4, epsilon=1.0, 
                              epsilon_decay=0.9, epsilon_min=0.1)
        
        initial_epsilon = agent.epsilon
        for _ in range(10):
            agent.decay_epsilon()
        
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_min
    
    def test_tie_breaking_in_best_action(self):
        """Test random tie-breaking when multiple actions have same Q-value"""
        agent = QLearningAgent(n_actions=4)
        
        state = (0, 0)
        # All actions have same Q-value (0.0)
        
        # Run multiple times, should see variation due to random tie-breaking
        actions = [agent.get_best_action(state) for _ in range(100)]
        unique_actions = len(set(actions))
        
        assert unique_actions > 1  # Should pick different actions


class TestTraining:
    """Test training dynamics and convergence"""
    
    def test_training_improves_performance(self):
        """Test agent performance improves with training"""
        env = GridWorld(size=5)
        agent = QLearningAgent(n_actions=4, epsilon=0.3, epsilon_decay=0.995)
        
        # Train for limited episodes
        stats = train_agent(env, agent, n_episodes=2000, verbose=False)
        
        # Later episodes should have higher rewards than early episodes
        early_rewards = np.mean(stats['episode_rewards'][:200])
        late_rewards = np.mean(stats['episode_rewards'][-200:])
        
        assert late_rewards > early_rewards
    
    def test_convergence_indicators(self):
        """Test Q-value changes decrease over time (convergence)"""
        env = GridWorld(size=5)
        agent = QLearningAgent(n_actions=4)
        
        stats = train_agent(env, agent, n_episodes=5000, verbose=False)
        
        # TD errors should decrease
        early_changes = np.mean(stats['q_value_changes'][:500])
        late_changes = np.mean(stats['q_value_changes'][-500:])
        
        assert late_changes < early_changes
    
    def test_q_table_population(self):
        """Test Q-table gets populated during training"""
        env = GridWorld(size=5)
        agent = QLearningAgent(n_actions=4)
        
        initial_size = len(agent.q_table)
        train_agent(env, agent, n_episodes=1000, verbose=False)
        final_size = len(agent.q_table)
        
        assert final_size > initial_size
        assert final_size > 20  # Should have explored many state-action pairs
    
    def test_learned_policy_quality(self):
        """Test learned policy successfully navigates to goal"""
        env = GridWorld(size=5)
        agent = QLearningAgent(n_actions=4, epsilon=0.2, epsilon_decay=0.995)
        
        # Train agent
        train_agent(env, agent, n_episodes=5000, verbose=False)
        
        # Test learned policy
        success_count = 0
        for _ in range(50):
            state = env.reset()
            steps = 0
            while steps < 20:
                action = agent.get_best_action(state)
                state, reward, done, _ = env.step(action)
                steps += 1
                if done and state == env.goal:
                    success_count += 1
                    break
                elif done:
                    break
        
        success_rate = success_count / 50
        assert success_rate > 0.7  # Should succeed at least 70% of the time
    
    def test_episode_statistics_tracking(self):
        """Test training statistics are properly recorded"""
        env = GridWorld(size=5)
        agent = QLearningAgent(n_actions=4)
        
        n_episodes = 500
        stats = train_agent(env, agent, n_episodes=n_episodes, verbose=False)
        
        assert len(stats['episode_rewards']) == n_episodes
        assert len(stats['episode_steps']) == n_episodes
        assert len(stats['epsilon_values']) == n_episodes
        assert len(stats['q_value_changes']) == n_episodes


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_small_grid(self):
        """Test 2x2 grid edge case"""
        env = GridWorld(size=2, obstacles=[])
        env.goal = (1, 1)
        agent = QLearningAgent(n_actions=4)
        
        # Should still train without errors
        train_agent(env, agent, n_episodes=100, verbose=False)
    
    def test_no_obstacles(self):
        """Test environment with no obstacles"""
        env = GridWorld(size=5, obstacles=[])
        agent = QLearningAgent(n_actions=4)
        
        stats = train_agent(env, agent, n_episodes=1000, verbose=False)
        
        # Should learn faster without obstacles
        final_rewards = np.mean(stats['episode_rewards'][-100:])
        assert final_rewards > 50
    
    def test_high_discount_factor(self):
        """Test agent with high discount factor (values future highly)"""
        env = GridWorld(size=5)
        agent = QLearningAgent(n_actions=4, discount_factor=0.99)
        
        train_agent(env, agent, n_episodes=2000, verbose=False)
        
        # Should still learn, might take longer
        assert len(agent.q_table) > 0
    
    def test_low_learning_rate(self):
        """Test agent with very low learning rate"""
        env = GridWorld(size=5)
        agent = QLearningAgent(n_actions=4, learning_rate=0.01)
        
        stats = train_agent(env, agent, n_episodes=2000, verbose=False)
        
        # Should still show some improvement, just slower
        assert len(stats['episode_rewards']) == 2000


def test_full_pipeline():
    """Integration test: complete training and evaluation pipeline"""
    # Create environment and agent
    env = GridWorld(size=5)
    agent = QLearningAgent(n_actions=4, learning_rate=0.1, 
                          discount_factor=0.95, epsilon=0.3)
    
    # Train
    stats = train_agent(env, agent, n_episodes=3000, verbose=False)
    
    # Verify training occurred
    assert len(agent.q_table) > 30
    assert len(stats['episode_rewards']) == 3000
    
    # Verify improvement
    early_performance = np.mean(stats['episode_rewards'][:300])
    late_performance = np.mean(stats['episode_rewards'][-300:])
    assert late_performance > early_performance
    
    # Extract policy
    policy = agent.get_policy()
    assert len(policy) > 0
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
