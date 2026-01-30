#!/bin/bash

# Day 100: Agents, Environments, and Rewards - Implementation Package Generator
# This script creates all necessary files for the RL fundamentals lesson

echo "Generating Day 100: Agents, Environments, and Rewards - Implementation Files..."

# Create requirements.txt
echo "Creating requirements.txt..."
cat > requirements.txt << 'EOF'
numpy>=1.21.0
gymnasium>=0.28.0
matplotlib>=3.5.0
pytest>=7.0.0
flask>=2.3.0
flask-cors>=4.0.0
EOF

# Create setup_venv.sh (venv setup script - different from this generator script)
echo "Creating setup_venv.sh..."
cat > setup_venv.sh << 'EOF'
#!/bin/bash

echo "Setting up Day 100: Agents, Environments, and Rewards Environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete! To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "Then run the lesson with:"
echo "python lesson_code.py"
echo ""
echo "Or run tests with:"
echo "pytest test_lesson.py -v"
EOF

chmod +x setup_venv.sh

# Create lesson_code.py
echo "Creating lesson_code.py..."
cat > lesson_code.py << 'EOF'
"""
Day 100: Agents, Environments, and Rewards
A comprehensive introduction to Reinforcement Learning fundamentals
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from collections import defaultdict


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
        
        if (episode + 1) % 10 == 0:
            stats = tracker.get_stats()
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Mean Reward: {stats['mean_reward']:.2f}, "
                  f"Last 10 Mean: {stats['last_10_mean']:.2f}")
    
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
EOF

# Create test_lesson.py
echo "Creating test_lesson.py..."
cat > test_lesson.py << 'EOF'
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
        env.state = (2, 2)  # Set to goal position
        state, reward, done, info = env.step(0)  # Any action
        assert done
        assert reward > 0


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
EOF

# Create startup script
echo "Creating startup.sh..."
cat > startup.sh << 'EOF'
#!/bin/bash

# Startup script for Day 100 RL Lesson
# This script starts the lesson and dashboard

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Day 100: Agents, Environments, and Rewards..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup..."
    if [ -f "setup_venv.sh" ]; then
        bash setup_venv.sh
    else
        echo "Error: setup_venv.sh not found"
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Check if dashboard script exists and start it
if [ -f "dashboard.py" ]; then
    echo "Starting dashboard..."
    python dashboard.py &
    DASHBOARD_PID=$!
    echo "Dashboard started with PID: $DASHBOARD_PID"
    sleep 2
fi

# Run the lesson
echo "Running lesson..."
python lesson_code.py

echo "Startup complete!"
EOF

chmod +x startup.sh

# Create dashboard.py
echo "Creating dashboard.py..."
cat > dashboard.py << 'EOF'
"""
Dashboard for Day 100: Agents, Environments, and Rewards
Real-time metrics visualization and monitoring
"""

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Metrics storage
metrics = {
    'episode_rewards': [],
    'episode_lengths': [],
    'total_episodes': 0,
    'mean_reward': 0.0,
    'mean_length': 0.0,
    'last_update': None
}


@app.route('/')
def index():
    """Dashboard home page"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Day 100 RL Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .metric-card {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #007bff;
            }
            .metric-label {
                color: #666;
                margin-top: 5px;
            }
            .chart-container {
                margin: 20px 0;
                height: 400px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Day 100: Agents, Environments, and Rewards - Dashboard</h1>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value" id="total-episodes">0</div>
                    <div class="metric-label">Total Episodes</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="mean-reward">0.00</div>
                    <div class="metric-label">Mean Reward</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="mean-length">0.00</div>
                    <div class="metric-label">Mean Episode Length</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="last-update">-</div>
                    <div class="metric-label">Last Update</div>
                </div>
            </div>
            
            <div class="chart-container">
                <canvas id="rewardsChart"></canvas>
            </div>
            
            <div class="chart-container">
                <canvas id="lengthsChart"></canvas>
            </div>
        </div>
        
        <script>
            const rewardsCtx = document.getElementById('rewardsChart').getContext('2d');
            const lengthsCtx = document.getElementById('lengthsChart').getContext('2d');
            
            const rewardsChart = new Chart(rewardsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Episode Rewards',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            const lengthsChart = new Chart(lengthsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Episode Lengths',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            function updateDashboard() {
                fetch('/api/metrics')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('total-episodes').textContent = data.total_episodes;
                        document.getElementById('mean-reward').textContent = data.mean_reward.toFixed(2);
                        document.getElementById('mean-length').textContent = data.mean_length.toFixed(2);
                        document.getElementById('last-update').textContent = 
                            data.last_update ? new Date(data.last_update).toLocaleTimeString() : '-';
                        
                        if (data.episode_rewards && data.episode_rewards.length > 0) {
                            const labels = data.episode_rewards.map((_, i) => i + 1);
                            rewardsChart.data.labels = labels;
                            rewardsChart.data.datasets[0].data = data.episode_rewards;
                            rewardsChart.update();
                        }
                        
                        if (data.episode_lengths && data.episode_lengths.length > 0) {
                            const labels = data.episode_lengths.map((_, i) => i + 1);
                            lengthsChart.data.labels = labels;
                            lengthsChart.data.datasets[0].data = data.episode_lengths;
                            lengthsChart.update();
                        }
                    })
                    .catch(error => console.error('Error fetching metrics:', error));
            }
            
            // Update every second
            setInterval(updateDashboard, 1000);
            updateDashboard();
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)


@app.route('/api/metrics')
def get_metrics():
    """Get current metrics"""
    # Try to load metrics from file if it exists
    metrics_file = 'metrics.json'
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                file_metrics = json.load(f)
                metrics.update(file_metrics)
        except:
            pass
    
    return jsonify(metrics)


@app.route('/api/update', methods=['POST'])
def update_metrics():
    """Update metrics (called by lesson code)"""
    # This would be called by the lesson code to update metrics
    # For now, we'll read from a metrics file
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("Starting Day 100 RL Dashboard on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
EOF

# Create sources directory and startup script there
echo "Creating sources directory and startup script..."
mkdir -p sources
cat > sources/startup.sh << 'EOF'
#!/bin/bash

# Startup script from sources directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Starting from sources directory..."
cd "$PROJECT_DIR"

if [ -f "startup.sh" ]; then
    bash "$PROJECT_DIR/startup.sh"
else
    echo "Error: startup.sh not found in project directory"
    exit 1
fi
EOF

chmod +x sources/startup.sh

echo ""
echo "All files generated successfully!"
echo ""
echo "Generated files:"
echo "  - requirements.txt"
echo "  - setup_venv.sh"
echo "  - lesson_code.py"
echo "  - test_lesson.py"
echo "  - startup.sh"
echo "  - dashboard.py"
echo "  - sources/startup.sh"
echo ""
echo "Next steps:"
echo "  1. Run: bash setup_venv.sh (or use setup.sh)"
echo "  2. Run: bash startup.sh"
echo "  3. Run tests: pytest test_lesson.py -v"
echo "  4. Access dashboard: http://localhost:5000"