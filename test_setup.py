"""
Quick test script to verify the setup works correctly
"""
import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import gymnasium as gym
        print("✓ gymnasium")
    except ImportError as e:
        print(f"✗ gymnasium: {e}")
        return False
    
    try:
        import torch
        print(f"✓ torch (version {torch.__version__})")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ numpy (version {np.__version__})")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ opencv-python (version {cv2.__version__})")
    except ImportError as e:
        print(f"✗ opencv-python: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ matplotlib (version {matplotlib.__version__})")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        from dqn_agent import DQNAgent, DQN_MLP, DQN_CNN, ReplayBuffer
        print("✓ dqn_agent module")
    except ImportError as e:
        print(f"✗ dqn_agent: {e}")
        return False
    
    try:
        from preprocessing import FramePreprocessor, FrameStack
        print("✓ preprocessing module")
    except ImportError as e:
        print(f"✗ preprocessing: {e}")
        return False
    
    try:
        from utils import plot_training_curve, plot_mountaincar_policy
        print("✓ utils module")
    except ImportError as e:
        print(f"✗ utils: {e}")
        return False
    
    return True


def test_environments():
    """Test if environments can be created"""
    print("\nTesting environments...")
    
    try:
        import gymnasium as gym
        
        # Test MountainCar
        env = gym.make("MountainCar-v0")
        obs, _ = env.reset()
        print(f"✓ MountainCar-v0")
        print(f"  State space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        env.close()
        
        # Test Pong
        try:
            env = gym.make("ALE/Pong-v5")
            obs, _ = env.reset()
            print(f"✓ ALE/Pong-v5")
            print(f"  State space: {env.observation_space}")
            print(f"  Action space: {env.action_space}")
            env.close()
        except Exception as e:
            print(f"✗ ALE/Pong-v5: {e}")
            print("  Try: pip install gymnasium[accept-rom-license]")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_agent_creation():
    """Test if agents can be created"""
    print("\nTesting agent creation...")
    
    try:
        from dqn_agent import DQNAgent
        
        # Test MLP agent
        agent = DQNAgent(
            state_dim=2,
            action_dim=3,
            network_type='mlp'
        )
        print("✓ MLP agent created")
        
        # Test CNN agent
        agent = DQNAgent(
            state_dim=4,
            action_dim=6,
            network_type='cnn'
        )
        print("✓ CNN agent created")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        return False


def test_preprocessing():
    """Test frame preprocessing"""
    print("\nTesting preprocessing...")
    
    try:
        import gymnasium as gym
        import numpy as np
        from preprocessing import FramePreprocessor, FrameStack
        
        # Create environment
        env = gym.make("ALE/Pong-v5")
        obs, _ = env.reset()
        
        # Test preprocessor
        preprocessor = FramePreprocessor(frame_size=(84, 84))
        processed = preprocessor.preprocess(obs)
        assert processed.shape == (84, 84), f"Expected (84, 84), got {processed.shape}"
        print(f"✓ Frame preprocessing (shape: {processed.shape})")
        
        # Test frame stack
        frame_stack = FrameStack(num_frames=4)
        stacked = frame_stack.reset(processed)
        assert stacked.shape == (4, 84, 84), f"Expected (4, 84, 84), got {stacked.shape}"
        print(f"✓ Frame stacking (shape: {stacked.shape})")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("DQN Implementation Setup Test")
    print("="*60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n⚠ Some imports failed. Install missing packages:")
        print("  pip install -r requirements.txt")
    
    # Test environments
    if not test_environments():
        all_passed = False
    
    # Test agent creation
    if not test_agent_creation():
        all_passed = False
    
    # Test preprocessing
    if not test_preprocessing():
        all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! Setup is ready.")
        print("\nYou can now:")
        print("  1. Open DQN_Training.ipynb in Jupyter")
        print("  2. Run the cells to train agents")
        print("  3. View results and visualizations")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()
