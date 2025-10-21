# DQN Implementation for MountainCar-v0 and Pong-v0

This project implements the Deep Q-Network (DQN) algorithm to solve two OpenAI Gym environments:
- **MountainCar-v0**: Classic control problem
- **Pong-v0**: Atari game with visual input

## Features

### Part A: Environment Exploration
- Load and inspect Gym environments
- Print state and action space information
- Test random agents to understand reward functions
- Analyze environment characteristics

### Part B: DQN Implementation
- **Experience Replay Buffer**: Stores and samples past experiences
- **Target Network**: Stabilizes training with periodic updates
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **Frame Preprocessing for Pong**:
  - RGB to grayscale conversion
  - Downsampling to 84×84 pixels
  - Frame stacking (4 frames) to capture motion
- **Neural Networks**:
  - MLP for MountainCar (simple feedforward network)
  - CNN for Pong (convolutional architecture)
- **Training Curves**: Visualize learning progress
- **Policy Visualization**: Plot action choices for MountainCar

## Project Structure

```
RL/
├── DQN_Training.ipynb      # Main Jupyter notebook with all experiments
├── dqn_agent.py            # DQN agent, networks, and replay buffer
├── preprocessing.py        # Frame preprocessing for Atari games
├── utils.py                # Visualization and utility functions
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### Google Colab Setup

Upload all `.py` files to Colab and run in the first cell:

```python
!pip install gymnasium[atari] gymnasium[accept-rom-license] ale-py torch torchvision opencv-python matplotlib
```

## Usage

### Running the Jupyter Notebook

1. Open `DQN_Training.ipynb` in Jupyter or Google Colab
2. Run cells sequentially to:
   - Explore environments
   - Test random agents
   - Train DQN on MountainCar-v0
   - Train DQN on Pong-v0
   - Visualize results

### Training Parameters

**MountainCar-v0:**
- Episodes: 500
- Training time: ~10-20 minutes on CPU
- Expected performance: -110 to -90 reward (solves in 90-110 steps)

**Pong-v0:**
- Episodes: 2000-4000 (2-4 million steps)
- Training time: Several hours (GPU recommended)
- Expected performance: -5 to +5 reward (competitive with opponent)

## Key Hyperparameters

### MountainCar-v0
```python
state_dim = 2
action_dim = 3
learning_rate = 1e-3
gamma = 0.99
epsilon_decay = 0.995
buffer_size = 50000
batch_size = 64
target_update_freq = 100
```

### Pong-v0
```python
state_dim = 4  # 4 stacked frames
action_dim = 6
learning_rate = 1e-4
gamma = 0.99
epsilon_decay = 0.9999
buffer_size = 100000
batch_size = 32
target_update_freq = 1000
```

## Results and Visualizations

The implementation generates:

1. **Training Curves** showing:
   - Episode rewards over time
   - Mean 100-episode reward
   - Episode lengths
   - Training loss

2. **MountainCar Policy Visualization**:
   - Action choices across position-velocity space
   - Q-value heatmap

3. **Performance Metrics**:
   - Mean reward with standard deviation
   - Best mean reward achieved
   - Episode lengths

## Observations

### MountainCar-v0

**Random Agent:**
- Mean reward: ~-200 (fails to reach goal)
- Strategy: Random actions don't build momentum

**Trained DQN Agent:**
- Mean reward: -110 to -90
- Strategy: Learns to oscillate to build momentum
- Policy: Pushes right when moving right, left when moving left

### Pong-v0

**Random Agent:**
- Mean reward: -21 to -15 (loses every point)
- Strategy: Random paddle movements

**Trained DQN Agent:**
- Early training (500-1000 episodes): Starts tracking ball
- Mid training (1000-2000 episodes): Wins occasional points (-15 to -10)
- Late training (2000+ episodes): Competitive player (-5 to +5)
- Strategy: Learns to position paddle to intercept ball

## Key Implementation Details

1. **Experience Replay**: Breaks correlation between consecutive samples, improves stability
2. **Target Network**: Prevents moving target problem during Q-learning
3. **Epsilon Decay**: Gradually shifts from exploration to exploitation
4. **Frame Preprocessing**: Reduces input dimensionality for Atari games
5. **Frame Stacking**: Captures temporal information (motion)
6. **Gradient Clipping**: Prevents exploding gradients

## Computational Requirements

**MountainCar-v0:**
- CPU sufficient
- ~10-20 minutes for 500 episodes
- Memory: ~1 GB

**Pong-v0:**
- GPU recommended (10-20x speedup)
- CPU: 6-12 hours for 2000 episodes
- GPU: 30-60 minutes for 2000 episodes
- Memory: ~2-4 GB

## Tips for Google Colab

1. **Enable GPU**: Runtime → Change runtime type → GPU
2. **Save checkpoints frequently**: Training can be interrupted
3. **Download results**: Save plots and checkpoints to Google Drive
4. **Monitor training**: Print progress every 10-20 episodes
5. **Start small**: Test with fewer episodes first

## Extending the Implementation

Possible improvements:
- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separate value and advantage streams
- **Prioritized Experience Replay**: Sample important transitions more frequently
- **Rainbow DQN**: Combines multiple improvements
- **Hyperparameter tuning**: Optimize learning rate, network architecture, etc.

## References

- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/)

## License

MIT License - Feel free to use and modify for your projects!

## Troubleshooting

**Issue**: ROM not found for Atari games
```bash
pip install gymnasium[accept-rom-license]
```

**Issue**: CUDA out of memory
- Reduce batch size
- Reduce buffer size
- Use CPU instead

**Issue**: Training is too slow
- Enable GPU in Colab
- Reduce number of episodes for testing
- Reduce frame stack size

**Issue**: Agent not learning
- Check epsilon decay (should decrease gradually)
- Verify reward scaling
- Increase training episodes
- Adjust learning rate
