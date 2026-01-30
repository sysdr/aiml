# Day 101: Q-Learning Algorithm

## Overview

A comprehensive implementation of Q-Learning, the foundational value-based reinforcement learning algorithm. This lesson demonstrates how agents learn optimal policies through trial-and-error interaction with their environment.

## What You'll Learn

- **Q-Table Mechanics**: Understanding value tables and state-action mappings
- **Bellman Equation**: The mathematical foundation of Q-Learning
- **Exploration vs Exploitation**: Epsilon-greedy strategies
- **Convergence Analysis**: How Q-values stabilize during training

## Quick Start

### Setup Environment
```bash
# Make setup script executable and run
chmod +x setup_environment.sh
./setup_environment.sh

# Activate virtual environment
source venv/bin/activate
```

### Train Q-Learning Agent
```bash
# Train for 10,000 episodes with visualizations
python lesson_code.py --episodes 10000 --visualize

# Expected output:
# - Training progress every 1000 episodes
# - Final success rate > 90%
# - Generated visualizations: gridworld_visualization.png, training_analysis.png
# - Saved Q-table: q_table.json
```

### Evaluate Trained Agent
```bash
# Test learned policy over 100 episodes
python lesson_code.py --test

# Expected output:
# - Success rate: ~95-100%
# - Average steps: ~8-10 (optimal path)
# - Consistent goal-reaching behavior
```

### Run Tests
```bash
# Execute comprehensive test suite
pytest test_lesson.py -v

# Expected: 20+ tests passing
# Tests cover:
# - Environment mechanics (Markov property, state transitions)
# - Agent learning (Q-value updates, epsilon-greedy)
# - Training dynamics (convergence, policy quality)
# - Edge cases (boundary conditions, parameter variations)
```

## Key Components

### GridWorld Environment
- **State Space**: 5×5 grid = 25 possible positions
- **Action Space**: {up, down, left, right}
- **Reward Structure**:
  - Goal: +100
  - Obstacle: -100
  - Step: -1 (encourages efficiency)

### Q-Learning Agent
- **Learning Rate (α)**: 0.1
- **Discount Factor (γ)**: 0.95
- **Epsilon**: 0.3 → 0.05 (with decay)
- **Q-Table**: Dictionary mapping (state, action) → Q-value

### Training Process
1. **Episode Start**: Agent at (0,0)
2. **Action Selection**: Epsilon-greedy policy
3. **Environment Step**: Observe reward and next state
4. **Q-Value Update**: Bellman equation
5. **Episode End**: Goal reached or obstacle hit
6. **Epsilon Decay**: Reduce exploration over time

## Algorithm Details

### Bellman Update Rule
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- **Q(s,a)**: Current Q-value estimate
- **α**: Learning rate (how much to update)
- **r**: Immediate reward
- **γ**: Discount factor (future reward importance)
- **max Q(s',a')**: Best Q-value from next state

### Epsilon-Greedy Policy
```python
if random() < epsilon:
    action = random_action()  # Explore
else:
    action = argmax Q(s,a)    # Exploit
```

## Expected Results

### Training Metrics
- **Episode 1-2000**: Random exploration, volatile rewards
- **Episode 2000-5000**: Learning phase, improving average reward
- **Episode 5000-10000**: Convergence, stable near-optimal policy

### Final Performance
- **Success Rate**: 95-100%
- **Average Reward**: ~95 (100 goal - 5 steps)
- **Average Steps**: 8-10 (close to optimal path length)
- **Q-Table Size**: 80-100 state-action pairs

## Visualizations

### 1. Grid World Visualization (`gridworld_visualization.png`)
- Color-coded states (red=obstacle, green=goal)
- Policy arrows showing learned behavior
- Clear visual of optimal path

### 2. Training Analysis (`training_analysis.png`)
Four subplots:
- **Episode Rewards**: Shows learning curve and convergence
- **Steps per Episode**: Efficiency improvement over time
- **Epsilon Decay**: Exploration-exploitation balance
- **Q-Value Changes**: Convergence indicator (decreasing TD errors)

## Real-World Applications

### Game Playing
- **AlphaGo**: Uses Deep Q-Networks (DQN) for board game mastery
- **Atari Games**: DQN learned to play 49 games from pixels alone

### Robotics
- **Boston Dynamics**: Q-Learning for motor control and balance
- **Warehouse Robots**: Path planning and task scheduling

### Resource Management
- **Google Data Centers**: 40% energy savings using RL for cooling
- **Telecommunications**: Network routing optimization

## Common Issues

### Agent Not Learning
- **Symptom**: Rewards stay constant
- **Solution**: Increase epsilon for more exploration or increase training episodes

### Slow Convergence
- **Symptom**: Takes >20,000 episodes
- **Solution**: Tune learning rate (try 0.15-0.2) or adjust discount factor

### Hitting Obstacles Repeatedly
- **Symptom**: Success rate <50%
- **Solution**: Check epsilon decay schedule, ensure sufficient exploration early

## Next Steps

- **Day 102**: Project Day - Implement complete RL agent for complex environment
- **Days 103-104**: Advanced RL concepts (eligibility traces, policy gradients)
- **Days 105-106**: Deep Q-Networks (DQN) - combining neural networks with Q-Learning

## Technical Notes

### Convergence Guarantees
Q-Learning converges to optimal Q* when:
1. All state-action pairs visited infinitely often
2. Learning rate α decreases appropriately
3. Environment satisfies Markov property

### Memory Efficiency
- Uses `defaultdict` instead of 2D array
- Only stores visited state-action pairs
- Scales to larger state spaces (10,000+ states)

### Computational Complexity
- **Per Update**: O(|A|) for max operation
- **Per Episode**: O(steps × |A|)
- **Total Training**: O(episodes × steps × |A|)

For our 5×5 grid: ~10,000 episodes × 20 steps × 4 actions = 800,000 operations (< 1 second)

## Dependencies

- `numpy>=1.26.4`: Numerical computing
- `matplotlib>=3.8.3`: Visualization
- `pytest>=8.0.2`: Testing framework
- `seaborn>=0.13.2`: Statistical plotting
- `tabulate>=0.9.0`: Table formatting

## File Structure

```
day_101_q_learning/
├── setup.sh              # Environment setup
├── requirements.txt      # Python dependencies
├── lesson_code.py        # Main implementation
├── test_lesson.py        # Test suite (20+ tests)
├── README.md            # This file
├── q_table.json         # Saved Q-values (after training)
└── *.png                # Generated visualizations
```

## Key Takeaways

1. **Q-Learning learns from experience**: No teacher needed, just rewards
2. **Bellman equation enables recursive optimization**: Future values bootstrap current values
3. **Exploration-exploitation trade-off is critical**: Balance learning new strategies with using known good ones
4. **Convergence requires coverage**: Must visit all state-action pairs sufficiently
5. **Foundation for modern RL**: DQN, Rainbow, MuZero all build on Q-Learning principles

---

**Author**: Day 101 of 180-Day AI/ML Course  
**Topics**: Reinforcement Learning, Value-Based Methods, Q-Learning  
**Difficulty**: Intermediate  
**Time**: 2-3 hours
