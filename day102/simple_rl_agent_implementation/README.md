# Day 102: Simple RL Agent - GridWorld Q-Learning

## Project Overview

Implementation of a reinforcement learning agent that learns to navigate a grid world using Q-Learning. This project demonstrates core RL concepts used in production AI systems at companies like DeepMind, OpenAI, and Amazon Robotics.

## What You'll Learn

- Agent-environment interaction loop
- Q-Learning algorithm implementation
- Exploration vs exploitation strategies
- Policy visualization and analysis
- Production RL system architecture patterns

## Quick Start

### 1. Setup Environment
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Run Training
```bash
python lesson_code.py
```

**Expected output:**
- Training progress printed every 100 episodes
- Final performance metrics
- Generated visualizations: `learned_policy.png`, `training_metrics.png`

### 3. Run Tests
```bash
pytest test_lesson.py -v
```

**Expected:** All 20+ tests pass

## Project Structure
```
.
├── lesson_code.py          # Main implementation (GridWorld, Agent, Trainer)
├── test_lesson.py          # Comprehensive test suite
├── requirements.txt        # Python dependencies
├── setup.sh               # Environment setup script
└── README.md              # This file
```

## Core Components

### GridWorld Environment
- 10×10 grid with obstacles
- State space: (x, y) coordinates
- Actions: UP, DOWN, LEFT, RIGHT
- Rewards: +10 (goal), -1 (obstacle), -0.1 (step)

### Q-Learning Agent
- Q-table for value function storage
- Epsilon-greedy exploration strategy
- Bellman equation updates
- Policy extraction

### Training Infrastructure
- Episode management
- Metrics tracking
- Convergence monitoring
- Visualization generation

## Hyperparameters
```python
learning_rate = 0.1       # How fast agent learns (α)
discount_factor = 0.99    # Future reward importance (γ)
epsilon_start = 1.0       # Initial exploration rate
epsilon_min = 0.01        # Minimum exploration
epsilon_decay = 0.995     # Decay per episode
```

## Experiments to Try

1. **Learning Rate Impact**
   - Try α = 0.01 (slow, stable) vs α = 0.5 (fast, unstable)

2. **Discount Factor**
   - Try γ = 0.5 (myopic) vs γ = 0.99 (far-sighted)

3. **Exploration Schedule**
   - Try different epsilon_decay values

4. **Environment Complexity**
   - Modify `obstacle_density` parameter
   - Change grid size

## Understanding the Outputs

### learned_policy.png
- **Left plot**: Value function heatmap (warmer = higher expected return)
- **Right plot**: Learned policy arrows showing optimal actions

### training_metrics.png
- **Top**: Episode rewards over time (should increase)
- **Middle**: Episode length (should decrease as agent learns)
- **Bottom**: Exploration rate decay

## Common Issues

**Problem:** Agent doesn't improve
- **Solution:** Increase training episodes or learning rate

**Problem:** Q-values explode to infinity
- **Solution:** Reduce learning rate, check reward calculation

**Problem:** Training too slow
- **Solution:** Reduce grid size, decrease obstacle density

## Real-World Connections

This implementation mirrors production RL systems:

- **Amazon Robotics**: Warehouse navigation with obstacle avoidance
- **Google DeepMind**: Game-playing agents (AlphaGo, AlphaStar)
- **Tesla Autopilot**: Lane change planning and path optimization
- **Netflix**: Exploration-exploitation in content recommendations

## Next Steps

Tomorrow (Day 103): Recommender Systems Theory
- Collaborative filtering
- Matrix factorization
- Hybrid approaches
- RL-based recommendations

## Resources

- Sutton & Barto: "Reinforcement Learning: An Introduction"
- David Silver's RL Course (DeepMind)
- OpenAI Spinning Up documentation

## Troubleshooting

**Import errors?**
```bash
pip install -r requirements.txt
```

**Tests failing?**
```bash
pytest test_lesson.py -v --tb=short
```

**Need help?** Check the inline code comments for detailed explanations.
