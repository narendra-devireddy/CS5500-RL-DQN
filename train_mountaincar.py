"""
Standalone script to train DQN on MountainCar-v0
Can be run directly without Jupyter notebook
"""
import gymnasium as gym
import numpy as np
import random
import torch

from dqn_agent import DQNAgent
from utils import plot_training_curve, plot_mountaincar_policy, save_results

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def train_dqn(env_name, agent, num_episodes=500, max_steps=200, 
              print_freq=10, save_freq=100, save_path='mountaincar_dqn.pth'):
    """Train DQN agent on MountainCar"""
    env = gym.make(env_name)
    
    episode_rewards = []
    episode_lengths = []
    losses = []
    mean_rewards = []
    best_mean_reward = -float('inf')
    
    total_steps = 0
    
    print(f"Training DQN on {env_name}...")
    print(f"Episodes: {num_episodes}, Max steps per episode: {max_steps}")
    print("-" * 80)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = obs
        
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take action
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = next_obs
            
            # Store transition
            agent.replay_buffer.push(state, action, next_state, reward, done or truncated)
            
            # Update agent
            loss = agent.update()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            total_steps += 1
            
            if done or truncated:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Calculate mean reward (last 100 episodes)
        if len(episode_rewards) >= 100:
            mean_reward = np.mean(episode_rewards[-100:])
            mean_rewards.append(mean_reward)
            
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                agent.save(save_path.replace('.pth', '_best.pth'))
                print(f"  → New best mean reward: {best_mean_reward:.2f}")
        
        # Print progress
        if (episode + 1) % print_freq == 0:
            mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Episode {episode+1:4d}/{num_episodes} | "
                  f"Steps: {total_steps:6d} | "
                  f"Reward: {episode_reward:6.2f} | "
                  f"Mean(100): {mean_reward:6.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {np.mean(episode_loss) if episode_loss else 0:.4f}")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(save_path)
            print(f"  → Checkpoint saved at episode {episode+1}")
    
    env.close()
    
    print("-" * 80)
    print(f"Training completed!")
    print(f"Best mean reward: {best_mean_reward:.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'losses': losses,
        'mean_rewards': mean_rewards,
        'best_mean_reward': best_mean_reward
    }


def test_agent(env_name, agent, num_episodes=10):
    """Test trained agent"""
    env = gym.make(env_name)
    
    test_rewards = []
    
    print(f"\nTesting agent on {env_name}...")
    print("-" * 80)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = obs
        
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated):
            action = agent.select_action(state, training=False)
            next_obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            state = next_obs
            steps += 1
        
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    env.close()
    
    print("-" * 80)
    print(f"Test Results:")
    print(f"  Mean Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Min Reward: {np.min(test_rewards):.2f}")
    print(f"  Max Reward: {np.max(test_rewards):.2f}")
    
    return test_rewards


def main():
    """Main training pipeline"""
    print("="*80)
    print("DQN Training on MountainCar-v0")
    print("="*80)
    
    # Create agent
    agent = DQNAgent(
        state_dim=2,
        action_dim=3,
        network_type='mlp',
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=50000,
        batch_size=64,
        target_update_freq=100
    )
    
    print(f"\nAgent Configuration:")
    print(f"  Network: MLP")
    print(f"  Learning rate: 1e-3")
    print(f"  Gamma: 0.99")
    print(f"  Epsilon: 1.0 → 0.01 (decay: 0.995)")
    print(f"  Buffer size: 50000")
    print(f"  Batch size: 64")
    print(f"  Target update frequency: 100")
    print()
    
    # Train agent
    results = train_dqn(
        env_name="MountainCar-v0",
        agent=agent,
        num_episodes=500,
        max_steps=200,
        print_freq=10,
        save_freq=100,
        save_path='mountaincar_dqn.pth'
    )
    
    # Save results
    save_results(results, 'mountaincar_results.pkl')
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curve(results, 'MountainCar-v0', window=50, save_path='mountaincar_training.png')
    
    # Plot policy
    print("Generating policy visualization...")
    plot_mountaincar_policy(agent, resolution=50)
    
    # Test agent
    test_rewards = test_agent("MountainCar-v0", agent, num_episodes=10)
    
    print("\n" + "="*80)
    print("Training pipeline completed successfully!")
    print("Generated files:")
    print("  - mountaincar_dqn.pth (checkpoint)")
    print("  - mountaincar_dqn_best.pth (best model)")
    print("  - mountaincar_results.pkl (training data)")
    print("  - mountaincar_training.png (training curves)")
    print("  - mountaincar_policy.png (policy visualization)")
    print("="*80)


if __name__ == "__main__":
    main()
