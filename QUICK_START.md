# 🚀 Quick Start Guide

## For Google Colab (Recommended)

### Step 1: Upload Files
Upload these files to Colab:
- ✅ `DQN_Training.ipynb`
- ✅ `dqn_agent.py`
- ✅ `preprocessing.py`
- ✅ `utils.py`

### Step 2: Enable GPU
- Click: **Runtime** → **Change runtime type** → **GPU**

### Step 3: Run the Notebook
- Open `DQN_Training.ipynb`
- Run cells sequentially (Shift+Enter)
- First cell installs dependencies (~2 minutes)

### Step 4: Train Agents
- **MountainCar**: Runs in ~15 minutes
- **Pong**: Runs in ~1-2 hours with GPU

---

## For Local Machine

### Step 1: Install Dependencies
```bash
cd /Users/ndi76jc/Desktop/development/RL
pip install -r requirements.txt
```

### Step 2: Test Setup
```bash
python test_setup.py
```

### Step 3: Choose Your Path

**Option A: Jupyter Notebook**
```bash
jupyter notebook DQN_Training.ipynb
```

**Option B: Standalone Script**
```bash
python train_mountaincar.py
```

---

## What You'll Get

### Part A: Environment Exploration
- ✓ State and action space information
- ✓ Random agent baseline performance
- ✓ Reward function analysis

### Part B: DQN Training
- ✓ Trained agents for both environments
- ✓ Training curves (rewards, loss, episode length)
- ✓ MountainCar policy visualization
- ✓ Model checkpoints (.pth files)

---

## Expected Results

### MountainCar-v0
```
Random Agent:  -200 reward (fails)
Trained DQN:    -95 reward (success!)
Training Time:  15 minutes
```

### Pong-v0
```
Random Agent:   -21 reward (loses badly)
After 1000 ep:  -12 reward (improving)
After 2000 ep:   -5 reward (competitive!)
Training Time:  1-2 hours (GPU) or 6-12 hours (CPU)
```

---

## Files Overview

| File | Purpose |
|------|---------|
| 📓 `DQN_Training.ipynb` | **START HERE** - Main notebook |
| 🤖 `dqn_agent.py` | DQN algorithm implementation |
| 🖼️ `preprocessing.py` | Frame preprocessing for Pong |
| 📊 `utils.py` | Plotting and visualization |
| 🧪 `test_setup.py` | Verify installation |
| 🏃 `train_mountaincar.py` | Standalone training script |
| 📋 `requirements.txt` | Python dependencies |
| 📖 `README.md` | Detailed documentation |
| 📘 `IMPLEMENTATION_GUIDE.md` | In-depth guide |

---

## Troubleshooting

### ❌ "ROM not found"
```bash
pip install gymnasium[accept-rom-license]
```

### ❌ "CUDA out of memory"
- Reduce batch_size in the notebook
- Or use CPU (slower but works)

### ❌ "Module not found"
```bash
pip install -r requirements.txt
```

### ❌ Training too slow
- Enable GPU in Colab
- Or reduce num_episodes for testing

---

## Next Steps

1. ✅ Run Part A to explore environments
2. ✅ Train MountainCar (quick, ~15 min)
3. ✅ Train Pong (longer, ~1-2 hours)
4. ✅ Analyze results and visualizations
5. 🎯 Experiment with hyperparameters
6. 🎯 Try other environments

---

## Need Help?

1. Check `test_setup.py` output
2. Read `IMPLEMENTATION_GUIDE.md`
3. Review error messages
4. Start with MountainCar first

**Happy Training! 🎮🤖**
