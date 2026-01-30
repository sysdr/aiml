"""
Day 100: Agents, Environments, and Rewards
A comprehensive introduction to Reinforcement Learning fundamentals
"""

import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from datetime import datetime


class SimpleEnvironment:
    """A simple grid world environment for RL demonstration"""
    
    def __init__(self, size=5):
        self.size = size
        self.state = None
        self.goal = (size - 1, size - 1)
        self.reset()
    
    def reset(self) -> Tuple[int, int]:
        """Reset environment to initial state"""
        self.state = (0, 0)
        return self.state
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """Take a step in the environment
        
        Actions: 0=up, 1=right, 2=down, 3=left
        """
        x, y = self.state
        reward = -0.1  # Small negative reward for each step
        
        if action == 0:  # Up
            y = max(0, y - 1)
        elif action == 1:  # Right
            x = min(self.size - 1, x + 1)
        elif action == 2:  # Down
            y = min(self.size - 1, y + 1)
        elif action == 3:  # Left
            x = max(0, x - 1)
        
        self.state = (x, y)
        
        # Check if goal reached
        done = self.state == self.goal
        if done:
            reward = 10.0  # Large reward for reaching goal
        
        info = {'state': self.state, 'goal': self.goal}
        return self.state, reward, done, info


class RandomAgent:
    """A simple random agent for baseline comparison"""
    
    def __init__(self, action_space_size: int):
        self.action_space_size = action_space_size
    
    def select_action(self, state) -> int:
        """Select a random action"""
        return np.random.randint(0, self.action_space_size)
    
    def update(self, state, action, reward, next_state, done):
        """Update agent (no-op for random agent)"""
        pass


class QLearningAgent:
    """A Q-Learning agent implementation"""
    
    def __init__(self, state_space_size: int, action_space_size: int, 
                 learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
    
    def state_to_key(self, state):
        """Convert state to hashable key"""
        return state
    
    def select_action(self, state) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space_size)
        else:
            state_key = self.state_to_key(state)
            return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning algorithm"""
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state_key])
        
        self.q_table[state_key][action] = current_q + self.learning_rate * (target_q - current_q)


class RewardTracker:
    """Track rewards and metrics during training"""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_rewards = []
    
    def record_episode(self, rewards: List[float], length: int):
        """Record an episode's metrics"""
        total_reward = sum(rewards)
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        self.total_rewards.append(total_reward)
    
    def get_stats(self) -> Dict:
        """Get statistics about training"""
        if not self.episode_rewards:
            return {}
        
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'total_episodes': len(self.episode_rewards),
            'last_10_mean': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
        }
    
    def save_metrics_to_file(self, filename='metrics.json'):
        """Save metrics to JSON file for dashboard"""
        stats = self.get_stats()
        metrics_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'total_episodes': len(self.episode_rewards),
            'mean_reward': float(stats.get('mean_reward', 0.0)),
            'mean_length': float(stats.get('mean_length', 0.0)),
            'last_update': datetime.now().isoformat()
        }
        with open(filename, 'w') as f:
            json.dump(metrics_data, f)
    
    def plot_metrics(self):
        """Plot training metrics"""
        if not self.episode_rewards:
            print("No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # Plot episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        print("Metrics saved to training_metrics.png")
        
        # Also save to JSON for dashboard
        self.save_metrics_to_file()


def train_agent(agent, environment, num_episodes=100):
    """Train an agent in the environment"""
    tracker = RewardTracker()
    
    for episode in range(num_episodes):
        state = environment.reset()
        episode_rewards = []
        done = False
        steps = 0
        
        while not done and steps < 1000:
            action = agent.select_action(state)
            next_state, reward, done, info = environment.step(action)
            agent.update(state, action, reward, next_state, done)
            
            episode_rewards.append(reward)
            state = next_state
            steps += 1
        
        tracker.record_episode(episode_rewards, steps)
        
        # Save metrics to file periodically for dashboard
        if (episode + 1) % 5 == 0:
            tracker.save_metrics_to_file()
        
        if (episode + 1) % 10 == 0:
            stats = tracker.get_stats()
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Mean Reward: {stats['mean_reward']:.2f}, "
                  f"Last 10 Mean: {stats['last_10_mean']:.2f}")
    
    # Final save
    tracker.save_metrics_to_file()
    return tracker


def main():
    """Main function to run the lesson"""
    print("=" * 60)
    print("Day 100: Agents, Environments, and Rewards")
    print("=" * 60)
    print()
    
    # Create environment
    print("Creating environment...")
    env = SimpleEnvironment(size=5)
    print(f"Environment created: {env.size}x{env.size} grid world")
    print(f"Goal position: {env.goal}")
    print()
    
    # Test random agent
    print("Testing Random Agent...")
    random_agent = RandomAgent(action_space_size=4)
    random_tracker = train_agent(random_agent, env, num_episodes=50)
    random_stats = random_tracker.get_stats()
    print(f"Random Agent - Mean Reward: {random_stats['mean_reward']:.2f}")
    print()
    
    # Train Q-Learning agent
    print("Training Q-Learning Agent...")
    q_agent = QLearningAgent(
        state_space_size=env.size * env.size,
        action_space_size=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    q_tracker = train_agent(q_agent, env, num_episodes=100)
    q_stats = q_tracker.get_stats()
    print(f"Q-Learning Agent - Mean Reward: {q_stats['mean_reward']:.2f}")
    print(f"Q-Learning Agent - Last 10 Episodes Mean: {q_stats['last_10_mean']:.2f}")
    print()
    
    # Plot metrics
    print("Generating metrics plots...")
    q_tracker.plot_metrics()
    print()
    
    # Display final statistics
    print("=" * 60)
    print("Final Statistics:")
    print("=" * 60)
    print(f"Random Agent:")
    print(f"  Mean Reward: {random_stats['mean_reward']:.2f} ± {random_stats['std_reward']:.2f}")
    print(f"  Mean Episode Length: {random_stats['mean_length']:.2f}")
    print()
    print(f"Q-Learning Agent:")
    print(f"  Mean Reward: {q_stats['mean_reward']:.2f} ± {q_stats['std_reward']:.2f}")
    print(f"  Mean Episode Length: {q_stats['mean_length']:.2f}")
    print(f"  Last 10 Episodes Mean: {q_stats['last_10_mean']:.2f}")
    print()
    
    print("Lesson complete! Check training_metrics.png for visualization.")


if __name__ == "__main__":
    main()
