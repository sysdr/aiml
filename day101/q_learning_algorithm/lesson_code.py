"""
Day 101: Q-Learning Algorithm Implementation
A complete Q-Learning agent that learns optimal policies in a Grid World environment
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
from collections import defaultdict
import argparse
import json


class GridWorld:
    """
    Grid World environment for Q-Learning demonstration
    
    State space: 2D grid positions (x, y)
    Action space: {0: up, 1: down, 2: left, 3: right}
    Reward structure: goal=+100, obstacle=-100, step=-1
    """
    
    def __init__(self, size: int = 5, obstacles: List[Tuple[int, int]] = None):
        self.size = size
        self.obstacles = obstacles or [(1, 1), (2, 3), (3, 1)]
        self.goal = (4, 4)
        self.start = (0, 0)
        self.state = self.start
        
        # Action mapping
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        self.action_names = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        self.n_actions = len(self.actions)
        
    def reset(self) -> Tuple[int, int]:
        """Reset environment to starting position"""
        self.state = self.start
        return self.state
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Take action and return (next_state, reward, done, info)
        
        Implements Markov property: next state depends only on current state and action
        """
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}. Must be in {list(self.actions.keys())}")
        
        # Calculate next position
        delta = self.actions[action]
        next_x = self.state[0] + delta[0]
        next_y = self.state[1] + delta[1]
        next_state = (next_x, next_y)
        
        # Check boundaries
        if not (0 <= next_x < self.size and 0 <= next_y < self.size):
            # Hit wall - stay in same position
            next_state = self.state
            reward = -1
            done = False
        elif next_state in self.obstacles:
            # Hit obstacle - large penalty
            reward = -100
            done = True
        elif next_state == self.goal:
            # Reached goal - large reward
            reward = 100
            done = True
        else:
            # Normal step - small penalty to encourage efficiency
            reward = -1
            done = False
        
        self.state = next_state
        info = {'position': next_state}
        
        return next_state, reward, done, info
    
    def get_state_representation(self) -> int:
        """Convert 2D position to 1D state index for Q-table"""
        return self.state[0] * self.size + self.state[1]
    
    def render(self, q_values: Optional[Dict] = None, policy: Optional[Dict] = None):
        """
        Visualize the grid world with optional Q-values and policy
        
        Args:
            q_values: Dictionary of (state, action) -> value
            policy: Dictionary of state -> best_action
        """
        grid = np.zeros((self.size, self.size))
        
        # Mark special positions
        grid[self.start] = 0.3
        grid[self.goal] = 1.0
        for obs in self.obstacles:
            grid[obs] = -1.0
        
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(grid, annot=False, cmap='RdYlGn', center=0, 
                    cbar_kws={'label': 'Value'}, vmin=-1, vmax=1)
        
        # Add policy arrows if provided
        if policy:
            for i in range(self.size):
                for j in range(self.size):
                    state = (i, j)
                    if state not in self.obstacles and state != self.goal:
                        if state in policy:
                            action = policy[state]
                            arrow = self.action_names[action]
                            plt.text(j + 0.5, i + 0.5, arrow, 
                                   ha='center', va='center', 
                                   fontsize=20, color='blue', weight='bold')
        
        # Add labels
        plt.text(self.start[1] + 0.5, self.start[0] + 0.5, 'S', 
                ha='center', va='center', fontsize=16, color='white', weight='bold')
        plt.text(self.goal[1] + 0.5, self.goal[0] + 0.5, 'G', 
                ha='center', va='center', fontsize=16, color='white', weight='bold')
        for obs in self.obstacles:
            plt.text(obs[1] + 0.5, obs[0] + 0.5, 'X', 
                    ha='center', va='center', fontsize=16, color='white', weight='bold')
        
        plt.title('Grid World Environment')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.tight_layout()
        plt.savefig('gridworld_visualization.png', dpi=150, bbox_inches='tight')
        print("✓ Grid visualization saved to gridworld_visualization.png")


class QLearningAgent:
    """
    Q-Learning Agent implementing tabular value-based reinforcement learning
    
    Core Algorithm:
    Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
    
    This is the Bellman optimality equation for Q-values
    """
    
    def __init__(self, 
                 n_actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent
        
        Args:
            n_actions: Number of possible actions
            learning_rate (α): How fast to update Q-values (0.1-0.3 typical)
            discount_factor (γ): How much to value future rewards (0.9-0.99 typical)
            epsilon (ε): Exploration rate for epsilon-greedy policy
            epsilon_decay: Rate at which epsilon decreases per episode
            epsilon_min: Minimum epsilon value
        """
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: maps (state, action) -> Q-value
        # Using defaultdict for memory efficiency with large state spaces
        self.q_table = defaultdict(float)
        
        # Statistics tracking
        self.training_stats = {
            'episode_rewards': [],
            'episode_steps': [],
            'epsilon_values': [],
            'q_value_changes': []
        }
        
    def get_q_value(self, state: Tuple[int, int], action: int) -> float:
        """Get Q-value for state-action pair"""
        return self.q_table.get((state, action), 0.0)
    
    def get_max_q_value(self, state: Tuple[int, int]) -> float:
        """Get maximum Q-value across all actions for given state"""
        return max([self.get_q_value(state, a) for a in range(self.n_actions)])
    
    def get_best_action(self, state: Tuple[int, int]) -> int:
        """Get action with highest Q-value (greedy policy)"""
        q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
        max_q = max(q_values)
        
        # Handle ties by random selection among best actions
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return np.random.choice(best_actions)
    
    def select_action(self, state: Tuple[int, int], training: bool = True) -> int:
        """
        Epsilon-greedy action selection
        
        With probability ε: random action (exploration)
        With probability 1-ε: best action (exploitation)
        
        This solves the exploration-exploitation trade-off
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit: best known action
            return self.get_best_action(state)
    
    def update(self, state: Tuple[int, int], action: int, 
               reward: float, next_state: Tuple[int, int], done: bool) -> float:
        """
        Update Q-value using Bellman equation
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Returns: magnitude of change (for convergence tracking)
        """
        current_q = self.get_q_value(state, action)
        
        if done:
            # Terminal state: no future rewards
            target = reward
        else:
            # Non-terminal: bootstrap from next state's max Q-value
            max_next_q = self.get_max_q_value(next_state)
            target = reward + self.gamma * max_next_q
        
        # Temporal difference (TD) error
        td_error = target - current_q
        
        # Q-value update
        new_q = current_q + self.alpha * td_error
        self.q_table[(state, action)] = new_q
        
        return abs(td_error)
    
    def decay_epsilon(self):
        """Decrease exploration rate over time"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_policy(self) -> Dict[Tuple[int, int], int]:
        """Extract greedy policy from Q-table"""
        policy = {}
        states = set(state for state, _ in self.q_table.keys())
        for state in states:
            policy[state] = self.get_best_action(state)
        return policy
    
    def save_q_table(self, filename: str):
        """Save Q-table to file"""
        # Convert tuple keys to strings for JSON serialization
        q_table_serializable = {
            f"{state[0]},{state[1]},{action}": value 
            for (state, action), value in self.q_table.items()
        }
        with open(filename, 'w') as f:
            json.dump({
                'q_table': q_table_serializable,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }, f, indent=2)
        print(f"✓ Q-table saved to {filename}")
    
    def load_q_table(self, filename: str):
        """Load Q-table from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert string keys back to tuples
        self.q_table = defaultdict(float)
        for key, value in data['q_table'].items():
            parts = key.split(',')
            state = (int(parts[0]), int(parts[1]))
            action = int(parts[2])
            self.q_table[(state, action)] = value
        
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        print(f"✓ Q-table loaded from {filename}")


def train_agent(env: GridWorld, agent: QLearningAgent, 
                n_episodes: int, verbose: bool = True) -> Dict:
    """
    Train Q-Learning agent through episodic interaction
    
    Training loop:
    1. Reset environment
    2. Select action using epsilon-greedy
    3. Take action, observe reward and next state
    4. Update Q-value using Bellman equation
    5. Repeat until episode ends
    6. Decay epsilon
    """
    print(f"\n🎓 Training Q-Learning agent for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        q_changes = []
        
        done = False
        while not done and episode_steps < 100:  # Max 100 steps per episode
            # Select and take action
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            
            # Update Q-value
            td_error = agent.update(state, action, reward, next_state, done)
            q_changes.append(td_error)
            
            # Accumulate statistics
            episode_reward += reward
            episode_steps += 1
            state = next_state
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Record statistics
        agent.training_stats['episode_rewards'].append(episode_reward)
        agent.training_stats['episode_steps'].append(episode_steps)
        agent.training_stats['epsilon_values'].append(agent.epsilon)
        agent.training_stats['q_value_changes'].append(np.mean(q_changes) if q_changes else 0)
        
        # Print progress
        if verbose and (episode + 1) % 1000 == 0:
            avg_reward = np.mean(agent.training_stats['episode_rewards'][-100:])
            avg_steps = np.mean(agent.training_stats['episode_steps'][-100:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Steps: {avg_steps:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("\n✓ Training complete!")
    return agent.training_stats


def evaluate_agent(env: GridWorld, agent: QLearningAgent, 
                   n_episodes: int = 100) -> Dict:
    """
    Evaluate trained agent performance
    
    Uses pure exploitation (epsilon=0) to test learned policy
    """
    print(f"\n🧪 Evaluating agent over {n_episodes} episodes...")
    
    success_count = 0
    total_rewards = []
    total_steps = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done and episode_steps < 100:
            # Pure exploitation: no exploration
            action = agent.select_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            # Check if reached goal
            if state == env.goal:
                success_count += 1
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
    
    results = {
        'success_rate': success_count / n_episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_steps': np.mean(total_steps),
        'std_reward': np.std(total_rewards),
        'std_steps': np.std(total_steps)
    }
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Success Rate: {results['success_rate']:.2%}")
    print(f"  Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Average Steps: {results['avg_steps']:.2f} ± {results['std_steps']:.2f}")
    
    return results


def visualize_learning(stats: Dict):
    """Create comprehensive training visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Episode Rewards over Time
    ax1 = axes[0, 0]
    rewards = stats['episode_rewards']
    window = 100
    smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
    ax1.plot(range(window-1, len(rewards)), smoothed_rewards, 
            color='red', linewidth=2, label=f'{window}-episode average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Learning Progress: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Steps per Episode
    ax2 = axes[0, 1]
    steps = stats['episode_steps']
    smoothed_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
    ax2.plot(steps, alpha=0.3, color='green', label='Raw')
    ax2.plot(range(window-1, len(steps)), smoothed_steps, 
            color='orange', linewidth=2, label=f'{window}-episode average')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Goal/Termination')
    ax2.set_title('Efficiency: Steps per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Epsilon Decay
    ax3 = axes[1, 0]
    ax3.plot(stats['epsilon_values'], color='purple', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon (Exploration Rate)')
    ax3.set_title('Exploration-Exploitation Balance')
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-value Changes (Convergence Indicator)
    ax4 = axes[1, 1]
    q_changes = stats['q_value_changes']
    smoothed_changes = np.convolve(q_changes, np.ones(window)/window, mode='valid')
    ax4.plot(range(window-1, len(q_changes)), smoothed_changes, 
            color='brown', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Avg TD Error Magnitude')
    ax4.set_title('Convergence: Q-value Updates')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Training analysis saved to training_analysis.png")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Day 101: Q-Learning Algorithm')
    parser.add_argument('--episodes', type=int, default=10000, 
                       help='Number of training episodes')
    parser.add_argument('--test', action='store_true', 
                       help='Evaluate pre-trained agent')
    parser.add_argument('--visualize', action='store_true', 
                       help='Generate visualizations')
    args = parser.parse_args()
    
    # Create environment
    env = GridWorld(size=5, obstacles=[(1, 1), (2, 3), (3, 1)])
    
    # Create agent
    agent = QLearningAgent(
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3,  # Start with high exploration
        epsilon_decay=0.995,
        epsilon_min=0.05
    )
    
    if args.test:
        # Load and evaluate pre-trained agent
        try:
            agent.load_q_table('q_table.json')
            evaluate_agent(env, agent, n_episodes=100)
            
            # Visualize learned policy
            policy = agent.get_policy()
            env.render(policy=policy)
        except FileNotFoundError:
            print("❌ No saved Q-table found. Train the agent first.")
            return
    else:
        # Train agent
        stats = train_agent(env, agent, n_episodes=args.episodes, verbose=True)
        
        # Evaluate trained agent
        results = evaluate_agent(env, agent, n_episodes=100)
        
        # Save Q-table
        agent.save_q_table('q_table.json')
        
        # Generate visualizations
        if args.visualize:
            visualize_learning(stats)
            policy = agent.get_policy()
            env.render(policy=policy)
        
        print("\n✅ Day 101 Complete!")
        print(f"   Final Success Rate: {results['success_rate']:.2%}")
        print(f"   Learned Q-values: {len(agent.q_table)} state-action pairs")


if __name__ == "__main__":
    main()
