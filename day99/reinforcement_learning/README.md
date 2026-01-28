# Day 99: Introduction to Reinforcement Learning

Learn the fundamentals of Reinforcement Learning by building a Q-Learning agent that navigates a grid world through trial and error.

## What You'll Learn

- Core RL concepts: agents, environments, states, actions, rewards
- Q-Learning algorithm and value functions
- Exploration vs. exploitation trade-off
- Training loops and convergence
- How RL powers production AI systems

## Quick Start

### 1. Setup Environment

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Run the Lesson

```bash
python lesson_code.py
```

Expected output:
- Training progress over 1000 episodes
- Final performance metrics
- Visualizations saved as PNG files

### 3. Run Tests

```bash
python -m pytest test_lesson.py -v
```

All 25+ tests should pass, validating:
- Environment dynamics
- Q-Learning updates
- Training convergence
- Policy learning

## What Gets Built

1. **GridWorld Environment** (5×5 grid)
   - Start: (0,0)
   - Goal: (4,4)
   - Obstacles to navigate around
   - Reward structure guides learning

2. **Q-Learning Agent**
   - Q-table stores state-action values
   - Epsilon-greedy exploration
   - Bellman equation updates
   - Policy extraction

3. **Training System**
   - 1000 episode training loop
   - Progress tracking and metrics
   - Visualization generation
   - Performance evaluation

## Output Files

- `learned_policy.png` - Arrows showing optimal action per state
- `training_progress.png` - Reward/length curves over training

## Production Connections

This implementation demonstrates core concepts used in:

- **OpenAI ChatGPT**: RLHF for alignment
- **Tesla FSD**: Continuous policy improvement
- **Google Data Centers**: Energy optimization
- **Netflix**: Recommendation exploration
- **Amazon Warehouses**: Robot coordination

## Key Concepts

### Q-Learning Update Rule

```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

- **α**: Learning rate (0.1) - how quickly we update beliefs
- **γ**: Discount factor (0.95) - value of future rewards
- **r**: Immediate reward
- **max(Q(s',a'))**: Best future value

### Epsilon-Greedy Strategy

```
With probability ε: explore (random action)
With probability 1-ε: exploit (best action)
```

Balances discovering new strategies vs. using known good ones.

## Experimentation Ideas

1. **Change grid size**: Try 10×10 or 20×20
2. **Adjust learning rate**: Test α=0.01 vs α=0.5
3. **Modify rewards**: Different penalty/goal values
4. **Add more obstacles**: Increase navigation difficulty
5. **Change exploration decay**: Faster/slower epsilon reduction

## Common Issues

**Q: Training seems slow**
- This is normal! RL learns through experience
- Try reducing episodes for faster testing
- Increase learning rate for quicker updates

**Q: Agent doesn't reach goal**
- Check epsilon decay - might need more exploration
- Verify obstacles aren't blocking all paths
- Increase training episodes

**Q: Visualizations not generated**
- Ensure matplotlib is installed
- Check write permissions in directory
- Verify training completed successfully

## Next Steps

Tomorrow (Day 100): Deep dive into **Agents, Environments, and Rewards**
- Designing effective reward functions
- Environment modeling strategies
- Agent architecture patterns
- Multi-agent systems

## Resources

- Article: `lesson_article.md` - Full conceptual explanation
- Code: `lesson_code.py` - Complete implementation
- Tests: `test_lesson.py` - Validation suite

## Time Estimate

- Setup: 5 minutes
- Running code: 3-5 minutes
- Understanding concepts: 30-45 minutes
- Experimentation: 1-2 hours
- **Total: 2-3 hours**

---

**Remember**: This same Q-Learning algorithm, scaled up and combined with neural networks (Deep Q-Learning), powers game-playing AI that surpasses human performance. You're learning production-grade RL fundamentals!

