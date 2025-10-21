# DQN Implementation Guide

## Quick Start

### Option 1: Jupyter Notebook (Recommended for Google Colab)

1. **Upload to Google Colab:**
   - Upload `DQN_Training.ipynb`, `dqn_agent.py`, `preprocessing.py`, and `utils.py`
   - Enable GPU: Runtime → Change runtime type → GPU

2. **Install dependencies:**
   ```python
   !pip install gymnasium[atari] gymnasium[accept-rom-license] ale-py torch torchvision opencv-python matplotlib
   ```

3. **Run the notebook cells sequentially**

### Option 2: Standalone Python Script (Local)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test setup:**
   ```bash
   python test_setup.py
   ```

3. **Train MountainCar:**
   ```bash
   python train_mountaincar.py
   ```

## Project Files

| File | Description |
|------|-------------|
| `DQN_Training.ipynb` | Main notebook with all experiments (Part A & B) |
| `dqn_agent.py` | DQN agent, neural networks, replay buffer |
| `preprocessing.py` | Frame preprocessing for Atari games |
| `utils.py` | Visualization and utility functions |
| `train_mountaincar.py` | Standalone script for MountainCar |
| `test_setup.py` | Verify installation and setup |
| `requirements.txt` | Python dependencies |
| `README.md` | Detailed documentation |

## Implementation Details

### Part A: Environment Exploration

The notebook includes functions to:
1. **Load environments** and print state/action space information
2. **Test random agents** to understand reward functions
3. **Analyze observations** about environment dynamics

**Key Findings:**
- **MountainCar**: Sparse rewards (-1 per step), requires momentum building
- **Pong**: Episodic rewards (+1/-1), requires visual processing

### Part B: DQN Algorithm

#### Core Components

1. **Replay Buffer** (`ReplayBuffer` class)
   - Stores transitions: (state, action, next_state, reward, done)
   - Breaks correlation between consecutive samples
   - Capacity: 50K (MountainCar), 100K (Pong)

2. **Neural Networks**
   - **MLP** (`DQN_MLP`): 2 hidden layers (128 units) for MountainCar
   - **CNN** (`DQN_CNN`): 3 conv layers + 2 FC layers for Pong
   - Architecture follows original DQN paper

3. **Target Network**
   - Separate network for computing target Q-values
   - Updated every N steps (100 for MountainCar, 1000 for Pong)
   - Stabilizes training

4. **Epsilon-Greedy Exploration**
   - Starts at ε=1.0 (full exploration)
   - Decays to ε=0.01-0.02 (mostly exploitation)
   - Decay rate: 0.995 (MountainCar), 0.9999 (Pong)

#### Frame Preprocessing (Pong)

```python
# Pipeline:
RGB (210×160×3) 
  → Grayscale (210×160) 
  → Resize (84×84) 
  → Normalize [0,1] 
  → Stack 4 frames (4×84×84)
```

**Benefits:**
- Reduces input size by ~95%
- Frame stacking captures motion
- Faster training, lower memory

#### Training Algorithm

```
For each episode:
  1. Reset environment, get initial state
  2. For each step:
     a. Select action (ε-greedy)
     b. Execute action, observe reward and next state
     c. Store transition in replay buffer
     d. Sample batch from buffer
     e. Compute loss: MSE(Q(s,a), r + γ·max Q'(s',a'))
     f. Update policy network
     g. Periodically update target network
  3. Decay epsilon
  4. Record statistics
```

## Training Parameters

### MountainCar-v0

```python
agent = DQNAgent(
    state_dim=2,           # [position, velocity]
    action_dim=3,          # [left, none, right]
    network_type='mlp',
    lr=1e-3,               # Learning rate
    gamma=0.99,            # Discount factor
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    buffer_size=50000,
    batch_size=64,
    target_update_freq=100
)

# Training
num_episodes = 500
max_steps = 200
training_time = ~15 minutes (CPU)
```

**Expected Performance:**
- Random agent: ~-200 reward
- Trained agent: -110 to -90 reward
- Solves in: 90-110 steps

### Pong-v0

```python
agent = DQNAgent(
    state_dim=4,           # 4 stacked frames
    action_dim=6,          # Pong actions
    network_type='cnn',
    lr=1e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.02,
    epsilon_decay=0.9999,  # Slower decay
    buffer_size=100000,
    batch_size=32,
    target_update_freq=1000
)

# Training
num_episodes = 2000-4000
max_steps = 10000
training_time = ~30-60 min (GPU), ~6-12 hours (CPU)
```

**Expected Performance:**
- Random agent: -21 to -15 reward
- After 1000 episodes: -15 to -10
- After 2000+ episodes: -5 to +5 (competitive)

## Visualizations

### 1. Training Curves
- **Episode Rewards**: Raw and moving average
- **Mean Reward**: 100-episode rolling mean
- **Episode Lengths**: Steps per episode
- **Training Loss**: MSE loss over time

### 2. MountainCar Policy Visualization
- **Action Map**: Shows which action (left/none/right) is chosen for each (position, velocity) state
- **Q-Value Map**: Heatmap of maximum Q-values across state space

**Interpretation:**
- Agent learns to push right when moving right (positive velocity)
- Agent learns to push left when moving left (negative velocity)
- Higher Q-values near the goal position

## Tips for Success

### MountainCar
✓ Trains quickly (15-20 min)
✓ Works well on CPU
✓ Good for testing setup
✓ Clear policy visualization

### Pong
⚠ Requires patience (hours of training)
⚠ GPU highly recommended
⚠ Start with fewer episodes to test
⚠ Monitor progress every 100 episodes

### Google Colab Tips
1. **Enable GPU** for Pong training
2. **Save checkpoints** frequently (every 100 episodes)
3. **Download results** to avoid losing progress
4. **Use smaller num_episodes** for initial testing
5. **Monitor GPU usage** to avoid disconnection

### Debugging
- **Agent not learning**: Check epsilon decay, increase episodes
- **Training unstable**: Reduce learning rate, increase target update frequency
- **Out of memory**: Reduce batch size or buffer size
- **Too slow**: Enable GPU, reduce frame stack size

## Expected Outputs

After running the complete notebook, you'll have:

### Files Generated
- `mountaincar_dqn.pth` - MountainCar checkpoint
- `mountaincar_dqn_best.pth` - Best MountainCar model
- `mountaincar_results.pkl` - Training statistics
- `mountaincar_training.png` - Training curves
- `mountaincar_policy.png` - Policy visualization
- `pong_dqn.pth` - Pong checkpoint
- `pong_dqn_best.pth` - Best Pong model
- `pong_results.pkl` - Training statistics
- `pong_training.png` - Training curves

### Plots
1. **Training Curves** (4 subplots):
   - Episode rewards with moving average
   - Mean 100-episode reward
   - Episode lengths
   - Training loss

2. **MountainCar Policy** (2 subplots):
   - Action choices across state space
   - Q-value heatmap

## Performance Benchmarks

### MountainCar-v0
| Metric | Random Agent | Trained DQN |
|--------|--------------|-------------|
| Mean Reward | -200 | -95 |
| Success Rate | 0% | ~100% |
| Steps to Goal | Never | 90-110 |
| Training Time | - | 15 min |

### Pong-v0
| Episodes | Mean Reward | Description |
|----------|-------------|-------------|
| 0 (Random) | -21 | Loses every point |
| 500 | -18 | Starts tracking ball |
| 1000 | -12 | Wins some points |
| 2000 | -5 | Competitive |
| 3000+ | 0 to +5 | Wins often |

## Common Issues & Solutions

### Issue: "ROM not found"
```bash
pip install gymnasium[accept-rom-license]
```

### Issue: "CUDA out of memory"
- Reduce `batch_size` to 16 or 32
- Reduce `buffer_size` to 50000
- Use CPU (slower but works)

### Issue: "Training too slow"
- Enable GPU in Colab
- Reduce `num_episodes` for testing
- Use smaller network

### Issue: "Agent not improving"
- Train longer (more episodes)
- Check epsilon is decaying
- Verify reward function
- Try different hyperparameters

## Next Steps

After completing this implementation, consider:

1. **Extensions**:
   - Double DQN
   - Dueling DQN
   - Prioritized Experience Replay
   - Rainbow DQN

2. **Other Environments**:
   - CartPole-v1
   - LunarLander-v2
   - Breakout-v0

3. **Hyperparameter Tuning**:
   - Grid search over learning rates
   - Different network architectures
   - Various exploration strategies

4. **Analysis**:
   - Compare with other algorithms (A3C, PPO)
   - Ablation studies
   - Transfer learning

## References

- **Original DQN Paper**: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- **Nature DQN**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- **Gymnasium Docs**: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
- **PyTorch Docs**: [https://pytorch.org/docs/](https://pytorch.org/docs/)

## Support

If you encounter issues:
1. Check `test_setup.py` output
2. Review error messages carefully
3. Verify all dependencies are installed
4. Check GPU availability for Pong
5. Start with MountainCar to verify setup

Good luck with your DQN implementation! 🚀
