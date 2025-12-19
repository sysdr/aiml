# Day 44: Simple Linear Regression Theory

## Quick Start

### Setup (2 minutes)
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Run Main Lesson (5 minutes)
```bash
# Test environment setup
python lesson_code.py --test-setup

# Train linear regression model
python lesson_code.py --train

# Run interactive demo
python lesson_code.py --demo

# Generate visualizations
python lesson_code.py --visualize
```

### Run Tests (1 minute)
```bash
pytest test_lesson.py -v
```

### Expected Outputs
- `regression_line.png` - Visual of fitted line through data
- `loss_curve.png` - Training loss over iterations
- `predictions.png` - Predicted vs actual values
- Console output showing w, b, and final MSE

## What You'll Learn
1. Mathematical foundation of linear regression
2. Gradient descent optimization from scratch
3. Loss function computation and monitoring
4. How to scale from theory to production systems

## Success Criteria
- [ ] Model converges (loss decreases smoothly)
- [ ] Final MSE < 5% of initial MSE
- [ ] All 15+ tests pass
- [ ] Can explain w and b in your own words
- [ ] Understand connection to neural networks

## Troubleshooting

**Loss increasing instead of decreasing?**
- Learning rate too high - reduce from 0.01 to 0.001

**Very slow convergence?**
- Learning rate too low - increase to 0.01 or 0.05
- May need more iterations

**Import errors?**
- Run `./setup.sh` again
- Ensure virtual environment is activated

## Next Steps
Tomorrow (Day 45): Implement same algorithm using scikit-learn in 5 lines of code!
