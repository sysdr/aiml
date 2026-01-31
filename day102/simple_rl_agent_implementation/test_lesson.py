"""
Day 102: Test Suite for Simple RL Agent

Comprehensive tests covering:
- Environment dynamics (state transitions, rewards, termination)
- Agent learning (Q-value updates, policy extraction)
- Training infrastructure (episode execution, convergence)
- Edge cases (invalid states, corner cases)
"""

import pytest
import numpy as np
from lesson_code import GridWorld, QLearningAgent, RLTrainer


class TestGridWorld:
    """Test environment implementation"""
    
    def test_environment_initialization(self):
        """Test grid creation and initial state"""
        env = GridWorld(size=5, obstacle_density=0.1)
        
        assert env.size == 5
        assert env.start_pos == (0, 0)
        assert env.goal_pos == (4, 4)
        assert env.grid.shape == (5, 5)
    
    def test_reset_returns_start_position(self):
        """Test environment reset"""
        env = GridWorld(size=5)
        state = env.reset()
        
        assert state == env.start_pos
        assert env.agent_pos == env.start_pos
    
    def test_valid_action_execution(self):
        """Test valid movement"""
        env = GridWorld(size=5, obstacle_density=0.0)  # No obstacles
        env.reset()
        
        # Move right
        next_state, reward, done = env.step(3)
        
        assert next_state == (0, 1)
        assert reward == -0.1  # Step penalty
        assert done is False
    
    def test_boundary_collision(self):
        """Test hitting grid boundaries"""
        env = GridWorld(size=5)
        env.reset()
        
        # Try to move left from (0,0) - should hit boundary
        next_state, reward, done = env.step(2)
        
        assert next_state == (0, 0)  # Stays in place
        assert reward == -1.0  # Collision penalty
        assert done is False
    
    def test_goal_reached(self):
        """Test goal detection"""
        env = GridWorld(size=5)
        env.agent_pos = (4, 3)  # One step from goal
        
        # Move right to goal
        next_state, reward, done = env.step(3)
        
        assert next_state == (4, 4)
        assert reward == 10.0  # Goal reward
        assert done is True
    
    def test_obstacle_collision(self):
        """Test obstacle detection"""
        env = GridWorld(size=5, obstacle_density=0.0)
        env.grid[1, 0] = -1  # Place obstacle
        env.reset()
        
        # Try to move down into obstacle
        next_state, reward, done = env.step(1)
        
        assert next_state == (0, 0)  # Stays in place
        assert reward == -1.0  # Collision penalty
    
    def test_get_valid_actions(self):
        """Test action masking"""
        env = GridWorld(size=5, obstacle_density=0.0)
        
        # Corner position - only 2 valid actions
        valid_actions = env.get_valid_actions((0, 0))
        assert 0 not in valid_actions  # Can't go up
        assert 2 not in valid_actions  # Can't go left
        assert 1 in valid_actions      # Can go down
        assert 3 in valid_actions      # Can go right


class TestQLearningAgent:
    """Test agent implementation"""
    
    def test_agent_initialization(self):
        """Test agent creation"""
        agent = QLearningAgent(
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon_start=1.0
        )
        
        assert agent.alpha == 0.1
        assert agent.gamma == 0.9
        assert agent.epsilon == 1.0
        assert len(agent.q_table) == 0
    
    def test_q_value_initialization(self):
        """Test Q-value defaults"""
        agent = QLearningAgent()
        
        # Unvisited state-actions should return 0
        q_val = agent.get_q_value((0, 0), 0)
        assert q_val == 0.0
    
    def test_epsilon_greedy_exploration(self):
        """Test exploration behavior"""
        agent = QLearningAgent(epsilon_start=1.0)
        
        # With epsilon=1.0, should always explore (random actions)
        actions = [agent.choose_action((0, 0), [0, 1, 2, 3]) 
                   for _ in range(100)]
        
        # Should see variety in actions (not all same)
        assert len(set(actions)) > 1
    
    def test_epsilon_greedy_exploitation(self):
        """Test exploitation behavior"""
        agent = QLearningAgent(epsilon_start=0.0)
        
        # Set Q-values manually
        agent.q_table[((0, 0), 0)] = 1.0
        agent.q_table[((0, 0), 1)] = 5.0  # Best action
        agent.q_table[((0, 0), 2)] = 2.0
        agent.q_table[((0, 0), 3)] = 3.0
        
        # With epsilon=0, should always exploit (choose action 1)
        actions = [agent.choose_action((0, 0), [0, 1, 2, 3]) 
                   for _ in range(20)]
        
        assert all(a == 1 for a in actions)
    
    def test_q_value_update(self):
        """Test Q-learning update rule"""
        agent = QLearningAgent(learning_rate=0.1, discount_factor=0.9)
        
        # Initialize Q-value
        state = (0, 0)
        action = 1
        agent.q_table[(state, action)] = 0.0
        
        # Perform update
        reward = 1.0
        next_state = (1, 0)
        agent.q_table[(next_state, 1)] = 5.0  # Next Q-value
        
        agent.update(state, action, reward, next_state, [0, 1, 2, 3])
        
        # Expected: Q = 0 + 0.1 * (1 + 0.9 * 5 - 0) = 0.55
        expected_q = 0.1 * (1.0 + 0.9 * 5.0)
        assert abs(agent.q_table[(state, action)] - expected_q) < 1e-6
    
    def test_epsilon_decay(self):
        """Test exploration decay"""
        agent = QLearningAgent(
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.9
        )
        
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_min
    
    def test_policy_extraction(self):
        """Test greedy policy"""
        agent = QLearningAgent()
        
        # Set Q-values
        state = (2, 2)
        agent.q_table[(state, 0)] = 1.0
        agent.q_table[(state, 1)] = 3.0
        agent.q_table[(state, 2)] = 2.0
        agent.q_table[(state, 3)] = 5.0  # Best
        
        # Policy should select action 3
        best_action = agent.get_policy(state, [0, 1, 2, 3])
        assert best_action == 3


class TestRLTrainer:
    """Test training infrastructure"""
    
    def test_trainer_initialization(self):
        """Test trainer creation"""
        env = GridWorld(size=5)
        agent = QLearningAgent()
        trainer = RLTrainer(env, agent)
        
        assert trainer.env is env
        assert trainer.agent is agent
        assert len(trainer.episode_rewards) == 0
    
    def test_episode_execution(self):
        """Test single episode run"""
        env = GridWorld(size=5, obstacle_density=0.0)
        agent = QLearningAgent(epsilon_start=0.5)
        trainer = RLTrainer(env, agent)
        
        reward, steps = trainer.run_episode()
        
        assert isinstance(reward, float)
        assert isinstance(steps, int)
        assert steps > 0
        assert steps <= 500  # Max episode length
    
    def test_training_loop(self):
        """Test multi-episode training"""
        env = GridWorld(size=5, obstacle_density=0.1)
        agent = QLearningAgent()
        trainer = RLTrainer(env, agent)
        
        trainer.train(num_episodes=50, eval_interval=25)
        
        assert len(trainer.episode_rewards) == 50
        assert len(trainer.episode_lengths) == 50
        assert len(trainer.epsilon_history) == 50
    
    def test_learning_progress(self):
        """Test that agent improves over time"""
        env = GridWorld(size=5, obstacle_density=0.1)
        agent = QLearningAgent()
        trainer = RLTrainer(env, agent)
        
        trainer.train(num_episodes=200, eval_interval=100)
        
        # Compare first 50 episodes to last 50 episodes
        early_reward = np.mean(trainer.episode_rewards[:50])
        late_reward = np.mean(trainer.episode_rewards[-50:])
        
        # Agent should improve (higher rewards later)
        assert late_reward > early_reward


class TestIntegration:
    """End-to-end integration tests"""
    
    def test_complete_training_pipeline(self):
        """Test full training workflow"""
        # Setup
        env = GridWorld(size=6, obstacle_density=0.15)
        agent = QLearningAgent()
        trainer = RLTrainer(env, agent)
        
        # Train
        trainer.train(num_episodes=100, eval_interval=50)
        
        # Verify Q-table populated
        assert len(agent.q_table) > 0
        
        # Verify metrics collected
        assert len(trainer.episode_rewards) == 100
        assert len(trainer.episode_lengths) == 100
    
    def test_deterministic_behavior(self):
        """Test reproducibility with fixed seeds"""
        import random
        
        # First run
        random.seed(42)
        np.random.seed(42)
        env1 = GridWorld(size=5)
        agent1 = QLearningAgent(epsilon_start=0.5)
        trainer1 = RLTrainer(env1, agent1)
        trainer1.train(num_episodes=50, eval_interval=50)
        
        # Second run with same seeds
        random.seed(42)
        np.random.seed(42)
        env2 = GridWorld(size=5)
        agent2 = QLearningAgent(epsilon_start=0.5)
        trainer2 = RLTrainer(env2, agent2)
        trainer2.train(num_episodes=50, eval_interval=50)
        
        # Results should be identical
        assert trainer1.episode_rewards == trainer2.episode_rewards


def test_imports():
    """Test that all necessary imports work"""
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import random
    
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
