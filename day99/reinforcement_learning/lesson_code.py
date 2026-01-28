"""
Day 99: Introduction to Reinforcement Learning
A Q-Learning agent that learns to navigate a grid world
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import time


class GridWorld:
    """
    Simple grid environment for RL training.
    Similar to environments used in robotics navigation and game AI.
    """
    
    def __init__(self, size: int = 5, obstacles: Optional[List[Tuple[int, int]]] = None):
        self.size = size
        self.obstacles = obstacles or [(1, 1), (2, 2), (3, 1)]
        self.goal = (size - 1, size - 1)
        self.start = (0, 0)
        self.current_state = self.start
        
        # Action mapping: 0=up, 1=right, 2=down, 3=left
        self.actions = ['↑', '→', '↓', '←']
        self.action_effects = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
    def reset(self) -> Tuple[int, int]:
        """Reset environment to start state."""
        self.current_state = self.start
        return self.current_state
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute action and return (next_state, reward, done).
        
        Production insight: This is similar to how Tesla's simulator
        models vehicle dynamics for training autonomous driving policies.
        """
        row, col = self.current_state
        d_row, d_col = self.action_effects[action]
        new_row, new_col = row + d_row, col + d_col
        
        # Check boundaries
        if not (0 <= new_row < self.size and 0 <= new_col < self.size):
            # Hit wall - stay in place with penalty
            return self.current_state, -10, False
        
        next_state = (new_row, new_col)
        
        # Check obstacles
        if next_state in self.obstacles:
            return self.current_state, -10, False
        
        # Check goal
        if next_state == self.goal:
            self.current_state = next_state
            return next_state, 100, True
        
        # Normal move
        self.current_state = next_state
        return next_state, -1, False
    
    def render(self):
        """Visualize the grid world."""
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:] = '.'
        
        for obs in self.obstacles:
            grid[obs] = 'X'
        
        grid[self.goal] = 'G'
        grid[self.current_state] = 'A'
        
        print("\nGrid World:")
        for row in grid:
            print(' '.join(row))
        print()


class QLearningAgent:
    """
    Q-Learning agent that learns optimal policies through experience.
    
    Production context: This algorithm powers recommendation systems,
    game AI, and robotic control across industry.
    """
    
    def __init__(
        self,
        state_space_size: int,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_space_size, state_space_size, action_space_size))
        
        self.alpha = learning_rate  # How quickly we update Q-values
        self.gamma = discount_factor  # How much we value future rewards
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.action_space_size = action_space_size
        
    def get_action(self, state: Tuple[int, int], training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Exploration vs. Exploitation trade-off:
        - Same principle Netflix uses to balance showing known favorites
          vs. discovering new content preferences
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.action_space_size)
        else:
            # Exploit: best known action
            row, col = state
            return np.argmax(self.q_table[row, col])
    
    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        done: bool
    ):
        """
        Update Q-value using the Q-learning equation.
        
        Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        
        This is the same update rule that powered DeepMind's early
        Atari game-playing agents.
        """
        row, col = state
        next_row, next_col = next_state
        
        # Current Q-value
        current_q = self.q_table[row, col, action]
        
        # Maximum future Q-value
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_row, next_col])
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[row, col, action] = new_q
    
    def decay_epsilon(self):
        """Gradually reduce exploration as agent learns."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class RLTrainer:
    """
    Training orchestrator for RL experiments.
    Tracks metrics and manages training loops.
    """
    
    def __init__(self, env: GridWorld, agent: QLearningAgent):
        self.env = env
        self.agent = agent
        self.episode_rewards = []
        self.episode_lengths = []
        
    def train(self, num_episodes: int = 1000, verbose: bool = True) -> dict:
        """
        Train agent for specified number of episodes.
        
        Production insight: Training loops like this run 24/7 in
        data centers, continuously improving AI policies based on
        real-world interaction data.
        """
        start_time = time.time()
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 100:  # Max 100 steps per episode
                # Choose and execute action
                action = self.agent.get_action(state, training=True)
                next_state, reward, done = self.env.step(action)
                
                # Learn from experience
                self.agent.update(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                steps += 1
            
            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            
            # Decay exploration
            self.agent.decay_epsilon()
            
            # Progress reporting
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.2f} | "
                      f"Epsilon: {self.agent.epsilon:.3f}")
        
        training_time = time.time() - start_time
        
        return {
            'episodes': num_episodes,
            'training_time': training_time,
            'final_avg_reward': np.mean(self.episode_rewards[-100:]),
            'final_avg_length': np.mean(self.episode_lengths[-100:])
        }
    
    def evaluate(self, num_episodes: int = 10) -> dict:
        """Evaluate trained agent without exploration."""
        test_rewards = []
        test_lengths = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 100:
                action = self.agent.get_action(state, training=False)
                next_state, reward, done = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                steps += 1
            
            test_rewards.append(episode_reward)
            test_lengths.append(steps)
        
        return {
            'avg_reward': np.mean(test_rewards),
            'avg_length': np.mean(test_lengths),
            'success_rate': sum(r > 0 for r in test_rewards) / num_episodes
        }
    
    def visualize_policy(self):
        """
        Visualize learned policy as arrows showing best action per state.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create grid
        grid = np.zeros((self.env.size, self.env.size))
        
        # Mark obstacles and goal
        for obs in self.env.obstacles:
            grid[obs] = -1
        grid[self.env.goal] = 1
        
        # Show grid
        im = ax.imshow(grid, cmap='RdYlGn', alpha=0.3)
        
        # Draw policy arrows
        for row in range(self.env.size):
            for col in range(self.env.size):
                if (row, col) in self.env.obstacles or (row, col) == self.env.goal:
                    continue
                
                # Get best action
                best_action = np.argmax(self.agent.q_table[row, col])
                arrow = self.env.actions[best_action]
                
                # Draw arrow
                ax.text(col, row, arrow, ha='center', va='center',
                       fontsize=20, color='blue', weight='bold')
        
        # Labels
        ax.set_xticks(range(self.env.size))
        ax.set_yticks(range(self.env.size))
        ax.set_title('Learned Policy (Arrows show best action per state)', 
                     fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('learned_policy.png', dpi=150, bbox_inches='tight')
        print("\nPolicy visualization saved to 'learned_policy.png'")
        plt.close()
    
    def plot_training_progress(self):
        """Plot reward and episode length over training."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Smooth rewards with moving average
        window = 50
        if len(self.episode_rewards) >= window:
            smoothed_rewards = np.convolve(
                self.episode_rewards, 
                np.ones(window)/window, 
                mode='valid'
            )
            ax1.plot(smoothed_rewards, linewidth=2, color='blue', label='Smoothed')
        
        ax1.plot(self.episode_rewards, alpha=0.3, color='lightblue', label='Raw')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Total Reward', fontsize=12)
        ax1.set_title('Training Progress: Rewards', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Episode lengths
        if len(self.episode_lengths) >= window:
            smoothed_lengths = np.convolve(
                self.episode_lengths,
                np.ones(window)/window,
                mode='valid'
            )
            ax2.plot(smoothed_lengths, linewidth=2, color='green', label='Smoothed')
        
        ax2.plot(self.episode_lengths, alpha=0.3, color='lightgreen', label='Raw')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Steps to Goal', fontsize=12)
        ax2.set_title('Training Progress: Episode Length', fontsize=14, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        print("Training progress plots saved to 'training_progress.png'")
        plt.close()


def demonstrate_rl_learning():
    """
    Main demonstration: Train RL agent and show results.
    """
    print("=" * 60)
    print("Day 99: Introduction to Reinforcement Learning")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating Grid World Environment...")
    env = GridWorld(size=5)
    env.render()
    
    # Create agent
    print("2. Initializing Q-Learning Agent...")
    agent = QLearningAgent(
        state_space_size=5,
        action_space_size=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    print(f"   Q-table shape: {agent.q_table.shape}")
    print(f"   Initial exploration rate: {agent.epsilon}")
    
    # Train agent
    print("\n3. Training Agent (1000 episodes)...")
    trainer = RLTrainer(env, agent)
    results = trainer.train(num_episodes=1000, verbose=True)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total episodes: {results['episodes']}")
    print(f"Training time: {results['training_time']:.2f} seconds")
    print(f"Final average reward: {results['final_avg_reward']:.2f}")
    print(f"Final average steps: {results['final_avg_length']:.2f}")
    
    # Evaluate
    print("\n4. Evaluating Trained Agent...")
    eval_results = trainer.evaluate(num_episodes=20)
    print(f"   Test average reward: {eval_results['avg_reward']:.2f}")
    print(f"   Test average steps: {eval_results['avg_length']:.2f}")
    print(f"   Success rate: {eval_results['success_rate']*100:.1f}%")
    
    # Visualize
    print("\n5. Generating Visualizations...")
    trainer.visualize_policy()
    trainer.plot_training_progress()
    
    # Demo optimal path
    print("\n6. Demonstrating Optimal Path:")
    state = env.reset()
    path = [state]
    steps = 0
    done = False
    
    while not done and steps < 20:
        action = agent.get_action(state, training=False)
        next_state, reward, done = env.step(action)
        path.append(next_state)
        state = next_state
        steps += 1
    
    print(f"   Path: {' → '.join([str(s) for s in path])}")
    print(f"   Steps: {steps}")
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("✓ Agent learned optimal navigation through trial and error")
    print("✓ No labeled examples needed - just reward signals")
    print("✓ Exploration-exploitation balance crucial for learning")
    print("✓ Same principles scale to production RL systems")
    print("\nCheck learned_policy.png and training_progress.png for visualizations!")


if __name__ == "__main__":
    demonstrate_rl_learning()

