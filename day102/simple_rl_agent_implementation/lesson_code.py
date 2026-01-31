"""
Day 102: Simple RL Agent - GridWorld Q-Learning Implementation

This module implements a complete reinforcement learning system with:
- GridWorld environment (state transitions, rewards)
- Q-Learning agent (value function, policy)
- Training infrastructure (episodes, metrics, visualization)

Architecture mirrors production RL systems at companies like:
- DeepMind (AlphaGo, AlphaStar)
- OpenAI (Dota 2, robotics)
- Amazon (warehouse robot navigation)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import random
from collections import defaultdict


class GridWorld:
    """
    GridWorld Environment - Discrete navigation task
    
    State space: (x, y) coordinates in NxN grid
    Action space: {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}
    Reward structure:
        - Goal reached: +10
        - Obstacle hit: -1
        - Each step: -0.1 (encourages shorter paths)
    
    Production analogy: Physics simulator for robot navigation,
    game engine for AI agents, market simulator for trading bots
    """
    
    def __init__(self, size: int = 10, obstacle_density: float = 0.2):
        """
        Initialize GridWorld environment
        
        Args:
            size: Grid dimensions (size x size)
            obstacle_density: Fraction of cells with obstacles
        """
        self.size = size
        self.obstacle_density = obstacle_density
        
        # Action mapping
        self.actions = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1)    # RIGHT
        }
        
        # Initialize grid layout
        self._create_grid()
        
    def _create_grid(self):
        """
        Create grid with obstacles, start, and goal positions
        
        Production consideration: In real systems, environments
        are dynamically loaded from databases or generated procedurally
        """
        # Initialize empty grid
        self.grid = np.zeros((self.size, self.size))
        
        # Place obstacles randomly (avoid start and goal)
        num_obstacles = int(self.size * self.size * self.obstacle_density)
        obstacles_placed = 0
        
        while obstacles_placed < num_obstacles:
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            # Avoid start (0,0) and goal (size-1, size-1)
            if (x, y) not in [(0, 0), (self.size-1, self.size-1)] and self.grid[x, y] == 0:
                self.grid[x, y] = -1  # Obstacle marker
                obstacles_placed += 1
        
        # Set start and goal
        self.start_pos = (0, 0)
        self.goal_pos = (self.size - 1, self.size - 1)
        self.grid[self.goal_pos] = 1  # Goal marker
        
    def reset(self) -> Tuple[int, int]:
        """
        Reset environment to start state
        
        Returns:
            Initial state (x, y)
        """
        self.agent_pos = self.start_pos
        return self.agent_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute action in environment, return next state
        
        Args:
            action: Integer in [0, 3] representing direction
            
        Returns:
            next_state: New (x, y) position
            reward: Reward received
            done: Whether episode terminated
            
        Production note: Real environments return additional info
        (e.g., sensor readings, collision flags, debug metadata)
        """
        # Calculate new position
        dx, dy = self.actions[action]
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy
        
        # Check boundaries
        if not (0 <= new_x < self.size and 0 <= new_y < self.size):
            # Hit wall - stay in place, negative reward
            return self.agent_pos, -1.0, False
        
        # Check obstacles
        if self.grid[new_x, new_y] == -1:
            # Hit obstacle - stay in place, negative reward
            return self.agent_pos, -1.0, False
        
        # Valid move - update position
        self.agent_pos = (new_x, new_y)
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 10.0, True  # Goal reached!
        else:
            return self.agent_pos, -0.1, False  # Small penalty per step
    
    def get_valid_actions(self, state: Tuple[int, int]) -> List[int]:
        """
        Get list of valid actions from current state
        
        Used in production for action masking - prevents agent
        from selecting physically impossible actions
        """
        valid = []
        for action in range(4):
            dx, dy = self.actions[action]
            new_x, new_y = state[0] + dx, state[1] + dy
            
            # Check if move is valid
            if (0 <= new_x < self.size and 0 <= new_y < self.size 
                and self.grid[new_x, new_y] != -1):
                valid.append(action)
        
        return valid if valid else [0]  # Always have at least one action


class QLearningAgent:
    """
    Q-Learning Agent - Model-free RL algorithm
    
    Learns action-value function Q(s,a) representing expected return
    when taking action a in state s and following optimal policy thereafter.
    
    Production analogy:
    - Netflix: Q(user_state, movie_id) = expected watch time
    - Robotics: Q(joint_angles, torque_command) = expected task completion
    - Trading: Q(market_state, trade_action) = expected profit
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        Initialize Q-Learning agent
        
        Args:
            learning_rate: How quickly agent updates Q-values (α)
            discount_factor: Importance of future rewards (γ)
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon per episode
        """
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table: dictionary mapping (state, action) -> Q-value
        # Production systems use neural networks instead for continuous states
        self.q_table = defaultdict(float)
        
    def get_q_value(self, state: Tuple[int, int], action: int) -> float:
        """
        Retrieve Q-value for state-action pair
        
        Returns 0.0 for unvisited state-actions (optimistic initialization)
        """
        return self.q_table[(state, action)]
    
    def choose_action(self, state: Tuple[int, int], valid_actions: List[int]) -> int:
        """
        Epsilon-greedy action selection
        
        With probability epsilon: explore (random action)
        With probability 1-epsilon: exploit (best known action)
        
        Production note: Real systems use sophisticated exploration like
        Thompson sampling, UCB, or curiosity-driven bonuses
        """
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(valid_actions)
        else:
            # Exploit: best action based on Q-values
            q_values = [self.get_q_value(state, a) for a in valid_actions]
            max_q = max(q_values)
            
            # Handle ties by random selection among best actions
            best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update(self, state: Tuple[int, int], action: int, 
               reward: float, next_state: Tuple[int, int], 
               next_valid_actions: List[int]):
        """
        Q-Learning update rule (Bellman equation)
        
        Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        
        This is the core of value-based RL. Update happens after each
        environment step in online learning. Production systems often
        use experience replay for sample efficiency.
        """
        # Current Q-value
        current_q = self.get_q_value(state, action)
        
        # Best next Q-value (max over valid actions)
        next_q_values = [self.get_q_value(next_state, a) for a in next_valid_actions]
        max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # TD target: r + γ·max Q(s',a')
        target = reward + self.gamma * max_next_q
        
        # TD error: target - current
        td_error = target - current_q
        
        # Update Q-value
        new_q = current_q + self.alpha * td_error
        self.q_table[(state, action)] = new_q
    
    def decay_epsilon(self):
        """
        Decay exploration rate after each episode
        
        Gradually shift from exploration to exploitation as agent learns
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_policy(self, state: Tuple[int, int], valid_actions: List[int]) -> int:
        """
        Extract greedy policy (best action) without exploration
        
        Used for evaluation and deployment after training
        """
        q_values = [self.get_q_value(state, a) for a in valid_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)


class RLTrainer:
    """
    Training Infrastructure - Orchestrates agent-environment interaction
    
    Handles:
    - Episode management (reset, rollout, termination)
    - Metrics tracking (rewards, episode length, convergence)
    - Checkpointing and early stopping
    - Policy visualization
    
    Production analogy: Ray RLlib trainer, Stable Baselines3 training loop
    """
    
    def __init__(self, env: GridWorld, agent: QLearningAgent):
        """
        Initialize trainer with environment and agent
        """
        self.env = env
        self.agent = agent
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        
    def run_episode(self) -> Tuple[float, int]:
        """
        Run single training episode
        
        Returns:
            total_reward: Sum of rewards in episode
            steps: Number of steps taken
        """
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done and steps < 500:  # Max 500 steps per episode
            # Get valid actions
            valid_actions = self.env.get_valid_actions(state)
            
            # Agent selects action
            action = self.agent.choose_action(state, valid_actions)
            
            # Environment executes action
            next_state, reward, done = self.env.step(action)
            
            # Get valid actions for next state
            next_valid_actions = self.env.get_valid_actions(next_state)
            
            # Agent learns from transition
            self.agent.update(state, action, reward, next_state, next_valid_actions)
            
            # Update for next iteration
            state = next_state
            total_reward += reward
            steps += 1
        
        return total_reward, steps
    
    def train(self, num_episodes: int = 1000, eval_interval: int = 100):
        """
        Main training loop
        
        Args:
            num_episodes: Total training episodes
            eval_interval: Print progress every N episodes
        """
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Grid size: {self.env.size}x{self.env.size}")
        print(f"Obstacle density: {self.env.obstacle_density:.2%}")
        print("-" * 60)
        
        for episode in range(1, num_episodes + 1):
            # Run episode
            total_reward, steps = self.run_episode()
            
            # Store metrics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            self.epsilon_history.append(self.agent.epsilon)
            
            # Decay exploration
            self.agent.decay_epsilon()
            
            # Print progress
            if episode % eval_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-eval_interval:])
                avg_steps = np.mean(self.episode_lengths[-eval_interval:])
                print(f"Episode {episode:4d}: "
                      f"Avg Reward = {avg_reward:6.2f}, "
                      f"Avg Steps = {avg_steps:4.1f}, "
                      f"Epsilon = {self.agent.epsilon:.2f}")
        
        print("-" * 60)
        print("Training completed!")
        print(f"Final average reward (last 100 episodes): "
              f"{np.mean(self.episode_rewards[-100:]):.2f}")
    
    def visualize_policy(self, filename: str = "learned_policy.png"):
        """
        Visualize learned Q-values and policy
        
        Creates heatmap showing:
        - Value function (max Q-value at each state)
        - Policy arrows (best action direction)
        """
        # Create value function grid
        value_grid = np.zeros((self.env.size, self.env.size))
        
        for i in range(self.env.size):
            for j in range(self.env.size):
                state = (i, j)
                if self.env.grid[i, j] == -1:  # Obstacle
                    value_grid[i, j] = np.nan
                else:
                    valid_actions = self.env.get_valid_actions(state)
                    q_values = [self.agent.get_q_value(state, a) for a in valid_actions]
                    value_grid[i, j] = max(q_values) if q_values else 0.0
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Value function heatmap
        im1 = ax1.imshow(value_grid, cmap='YlOrRd', interpolation='nearest')
        ax1.set_title("Value Function (Max Q-value)", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Y Coordinate")
        ax1.set_ylabel("X Coordinate")
        
        # Mark start and goal
        ax1.plot(self.env.start_pos[1], self.env.start_pos[0], 
                'gs', markersize=15, label='Start')
        ax1.plot(self.env.goal_pos[1], self.env.goal_pos[0], 
                'b*', markersize=20, label='Goal')
        ax1.legend()
        plt.colorbar(im1, ax=ax1, label='Expected Return')
        
        # Plot 2: Policy arrows
        ax2.imshow(value_grid, cmap='YlOrRd', interpolation='nearest', alpha=0.3)
        ax2.set_title("Learned Policy (Best Actions)", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Y Coordinate")
        ax2.set_ylabel("X Coordinate")
        
        # Draw policy arrows
        arrow_mapping = {
            0: (0, -0.3),   # UP
            1: (0, 0.3),    # DOWN
            2: (-0.3, 0),   # LEFT
            3: (0.3, 0)     # RIGHT
        }
        
        for i in range(self.env.size):
            for j in range(self.env.size):
                state = (i, j)
                if self.env.grid[i, j] != -1:  # Not obstacle
                    valid_actions = self.env.get_valid_actions(state)
                    best_action = self.agent.get_policy(state, valid_actions)
                    dx, dy = arrow_mapping[best_action]
                    ax2.arrow(j, i, dy, dx, head_width=0.2, head_length=0.2, 
                             fc='blue', ec='blue', alpha=0.7)
        
        # Mark start and goal
        ax2.plot(self.env.start_pos[1], self.env.start_pos[0], 
                'gs', markersize=15, label='Start')
        ax2.plot(self.env.goal_pos[1], self.env.goal_pos[0], 
                'b*', markersize=20, label='Goal')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPolicy visualization saved to '{filename}'")
    
    def plot_training_metrics(self, filename: str = "training_metrics.png"):
        """
        Plot training progress over episodes
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Smooth curves with moving average
        window = 50
        
        # Plot 1: Episode rewards
        ax1.plot(self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) >= window:
            smoothed = np.convolve(self.episode_rewards, 
                                  np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(self.episode_rewards)), 
                    smoothed, linewidth=2, label='Smoothed')
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Training Rewards Over Time", fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Episode lengths
        ax2.plot(self.episode_lengths, alpha=0.3, label='Raw')
        if len(self.episode_lengths) >= window:
            smoothed = np.convolve(self.episode_lengths, 
                                  np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(self.episode_lengths)), 
                    smoothed, linewidth=2, label='Smoothed')
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps to Goal")
        ax2.set_title("Episode Length (Efficiency)", fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Exploration rate
        ax3.plot(self.epsilon_history, linewidth=2, color='orange')
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Epsilon (Exploration Rate)")
        ax3.set_title("Exploration vs Exploitation", fontweight='bold')
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Training metrics saved to '{filename}'")


def main():
    """
    Main execution: Create environment, agent, train, visualize
    """
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create environment
    env = GridWorld(size=10, obstacle_density=0.2)
    
    # Create agent with carefully tuned hyperparameters
    agent = QLearningAgent(
        learning_rate=0.1,       # How fast agent learns
        discount_factor=0.99,    # How much agent values future rewards
        epsilon_start=1.0,       # Start with full exploration
        epsilon_min=0.01,        # End with 1% exploration
        epsilon_decay=0.995      # Decay rate per episode
    )
    
    # Create trainer
    trainer = RLTrainer(env, agent)
    
    # Train agent
    trainer.train(num_episodes=1000, eval_interval=100)
    
    # Visualize results
    trainer.visualize_policy()
    trainer.plot_training_metrics()
    
    print("\n" + "="*60)
    print("SUCCESS! Your RL agent learned to navigate the GridWorld!")
    print("="*60)
    print("\nNext steps:")
    print("1. Open 'learned_policy.png' to see the learned value function")
    print("2. Open 'training_metrics.png' to analyze training progress")
    print("3. Experiment with hyperparameters in the code")
    print("4. Try different grid sizes and obstacle densities")


if __name__ == "__main__":
    main()
