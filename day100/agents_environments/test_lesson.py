"""
Tests for Day 100: Agents, Environments, and Rewards
"""

import pytest
import numpy as np
from lesson_code import (
    SimpleEnvironment,
    RandomAgent,
    QLearningAgent,
    RewardTracker,
    train_agent
)


class TestSimpleEnvironment:
    """Test the SimpleEnvironment class"""
    
    def test_environment_initialization(self):
        """Test environment is initialized correctly"""
        env = SimpleEnvironment(size=5)
        assert env.size == 5
        assert env.goal == (4, 4)
    
    def test_environment_reset(self):
        """Test environment reset"""
        env = SimpleEnvironment(size=5)
        state = env.reset()
        assert state == (0, 0)
        assert env.state == (0, 0)
    
    def test_environment_step(self):
        """Test environment step function"""
        env = SimpleEnvironment(size=5)
        env.reset()
        
        # Test moving right
        state, reward, done, info = env.step(1)
        assert state == (1, 0)
        assert not done
        assert 'state' in info
    
    def test_environment_goal_reached(self):
        """Test goal detection"""
        env = SimpleEnvironment(size=3)
        env.reset()
        # Move to just before goal
        env.state = (2, 1)
        # Move right to reach goal
        state, reward, done, info = env.step(2)  # Move down to (2, 2)
        assert done
        assert reward > 0
        assert state == env.goal


class TestRandomAgent:
    """Test the RandomAgent class"""
    
    def test_random_agent_initialization(self):
        """Test random agent initialization"""
        agent = RandomAgent(action_space_size=4)
        assert agent.action_space_size == 4
    
    def test_random_agent_action_selection(self):
        """Test random agent selects valid actions"""
        agent = RandomAgent(action_space_size=4)
        for _ in range(100):
            action = agent.select_action(None)
            assert 0 <= action < 4


class TestQLearningAgent:
    """Test the QLearningAgent class"""
    
    def test_qlearning_agent_initialization(self):
        """Test Q-learning agent initialization"""
        agent = QLearningAgent(state_space_size=25, action_space_size=4)
        assert agent.action_space_size == 4
        assert agent.learning_rate == 0.1
        assert agent.discount_factor == 0.95
    
    def test_qlearning_agent_action_selection(self):
        """Test Q-learning agent selects valid actions"""
        agent = QLearningAgent(state_space_size=25, action_space_size=4)
        for _ in range(100):
            action = agent.select_action((0, 0))
            assert 0 <= action < 4
    
    def test_qlearning_agent_update(self):
        """Test Q-learning agent update"""
        agent = QLearningAgent(state_space_size=25, action_space_size=4)
        state = (0, 0)
        action = 1
        reward = 1.0
        next_state = (1, 0)
        done = False
        
        # Initial Q-value should be 0
        state_key = agent.state_to_key(state)
        initial_q = agent.q_table[state_key][action]
        
        # Update agent
        agent.update(state, action, reward, next_state, done)
        
        # Q-value should have changed
        updated_q = agent.q_table[state_key][action]
        assert updated_q != initial_q
        assert updated_q > 0


class TestRewardTracker:
    """Test the RewardTracker class"""
    
    def test_reward_tracker_initialization(self):
        """Test reward tracker initialization"""
        tracker = RewardTracker()
        assert len(tracker.episode_rewards) == 0
    
    def test_reward_tracker_record_episode(self):
        """Test recording episodes"""
        tracker = RewardTracker()
        tracker.record_episode([1.0, 2.0, 3.0], 3)
        assert len(tracker.episode_rewards) == 1
        assert tracker.episode_rewards[0] == 6.0
        assert tracker.episode_lengths[0] == 3
    
    def test_reward_tracker_stats(self):
        """Test getting statistics"""
        tracker = RewardTracker()
        tracker.record_episode([1.0, 2.0], 2)
        tracker.record_episode([3.0, 4.0], 2)
        
        stats = tracker.get_stats()
        assert 'mean_reward' in stats
        assert 'mean_length' in stats
        assert stats['total_episodes'] == 2
        assert stats['mean_reward'] == 5.0


class TestTraining:
    """Test training functions"""
    
    def test_train_agent(self):
        """Test training an agent"""
        env = SimpleEnvironment(size=3)
        agent = RandomAgent(action_space_size=4)
        tracker = train_agent(agent, env, num_episodes=5)
        
        assert tracker is not None
        stats = tracker.get_stats()
        assert stats['total_episodes'] == 5
        assert 'mean_reward' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
