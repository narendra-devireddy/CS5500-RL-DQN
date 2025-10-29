"""
Command-line training script for Policy Gradient
Supports various configurations and environments
"""

import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pg_agent import PolicyGradientAgent, device


def train_policy_gradient(
    env_name,
    num_iterations,
    batch_size,
    lr=3e-4,
    gamma=0.99,
    use_reward_to_go=True,
    use_advantage_normalization=True,
    use_baseline=True,
    hidden_sizes=[64, 64],
    max_episode_length=1000,
    print_freq=10,
    save_path=None,
    seed=42
):
    """
    Train policy gradient agent
    
    Args:
        env_name: Name of Gym environment
        num_iterations: Number of training iterations
        batch_size: Number of timesteps to collect per iteration
        lr: Learning rate
        gamma: Discount factor
        use_reward_to_go: Use reward-to-go formulation
        use_advantage_normalization: Normalize advantages
        use_baseline: Use value function baseline
        hidden_sizes: Hidden layer sizes
        max_episode_length: Maximum episode length
        print_freq: Print frequency
        save_path: Path to save results
        seed: Random seed
    
    Returns:
        results: Dictionary containing training statistics
    """
    # Set random seeds
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    
    # Create environment
    env = gym.make(env_name)
    env.reset(seed=seed)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"\n{'='*80}")
    print(f"Training Policy Gradient on {env_name}")
    print(f"{'='*80}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Reward-to-go: {use_reward_to_go}")
    print(f"Advantage normalization: {use_advantage_normalization}")
    print(f"Baseline: {use_baseline}")
    print(f"Learning rate: {lr}")
    print(f"Gamma: {gamma}")
    print(f"Batch size: {batch_size}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Create agent
    agent = PolicyGradientAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        use_reward_to_go=use_reward_to_go,
        use_advantage_normalization=use_advantage_normalization,
        use_baseline=use_baseline,
        hidden_sizes=hidden_sizes
    )
    
    # Training statistics
    iteration_returns = []
    iteration_lengths = []
    policy_losses = []
    value_losses = []
    
    total_timesteps = 0
    
    for iteration in range(num_iterations):
        # Collect batch of trajectories
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_returns = []
        batch_lengths = []
        
        timesteps_this_iteration = 0
        
        while timesteps_this_iteration < batch_size:
            # Reset trajectory storage
            agent.reset_trajectories()
            
            # Run episode
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_episode_length):
                # Select action
                action = agent.select_action(state, training=True)
                
                # Take action
                next_state, reward, done, truncated, info = env.step(action)
                
                # Store reward
                agent.store_reward(reward)
                
                episode_reward += reward
                episode_length += 1
                timesteps_this_iteration += 1
                
                state = next_state
                
                if done or truncated:
                    break
            
            # Store episode statistics
            batch_returns.append(episode_reward)
            batch_lengths.append(episode_length)
            
            # Update agent after each episode
            policy_loss, value_loss = agent.update()
            
            if policy_loss is not None:
                policy_losses.append(policy_loss)
            if value_loss is not None:
                value_losses.append(value_loss)
        
        # Record iteration statistics
        mean_return = np.mean(batch_returns)
        mean_length = np.mean(batch_lengths)
        iteration_returns.append(mean_return)
        iteration_lengths.append(mean_length)
        
        total_timesteps += timesteps_this_iteration
        
        # Print progress
        if (iteration + 1) % print_freq == 0:
            print(f"Iteration {iteration+1}/{num_iterations} | "
                  f"Timesteps: {total_timesteps} | "
                  f"Mean Return: {mean_return:.2f} | "
                  f"Mean Length: {mean_length:.2f} | "
                  f"Policy Loss: {np.mean(policy_losses[-10:]) if policy_losses else 0:.4f}")
    
    env.close()
    
    # Prepare results
    results = {
        'iteration_returns': iteration_returns,
        'iteration_lengths': iteration_lengths,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'policy_state_dict': agent.policy_net.state_dict(),
        'value_state_dict': agent.value_net.state_dict() if hasattr(agent, 'value_net') else None,
        'config': {
            'env_name': env_name,
            'num_iterations': num_iterations,
            'batch_size': batch_size,
            'lr': lr,
            'gamma': gamma,
            'use_reward_to_go': use_reward_to_go,
            'use_advantage_normalization': use_advantage_normalization,
            'use_baseline': use_baseline,
            'hidden_sizes': hidden_sizes,
            'seed': seed
        }
    }
    
    # Save results
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {save_path}")
    
    return agent, results


def main():
    parser = argparse.ArgumentParser(description='Train Policy Gradient Agent')
    
    # Environment
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Gym environment name (default: CartPole-v1, use LunarLander-v3)')
    
    # Training parameters
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of training iterations (default: 100)')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='Batch size (timesteps per iteration) (default: 5000)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    
    # Variance reduction techniques
    parser.add_argument('--reward_to_go', action='store_true', default=True,
                        help='Use reward-to-go (default: True)')
    parser.add_argument('--no_reward_to_go', dest='reward_to_go', action='store_false',
                        help='Do not use reward-to-go')
    parser.add_argument('--advantage_norm', action='store_true', default=True,
                        help='Use advantage normalization (default: True)')
    parser.add_argument('--no_advantage_norm', dest='advantage_norm', action='store_false',
                        help='Do not use advantage normalization')
    parser.add_argument('--baseline', action='store_true', default=True,
                        help='Use value function baseline (default: True)')
    parser.add_argument('--no_baseline', dest='baseline', action='store_false',
                        help='Do not use baseline')
    
    # Network architecture
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[64, 64],
                        help='Hidden layer sizes (default: 64 64)')
    
    # Other parameters
    parser.add_argument('--max_episode_length', type=int, default=1000,
                        help='Maximum episode length (default: 1000)')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save results (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Auto-generate save path if not provided
    if args.save_path is None:
        config_str = f"rtg{int(args.reward_to_go)}_norm{int(args.advantage_norm)}_base{int(args.baseline)}"
        args.save_path = f"pg_results_{args.env}_{config_str}.pkl"
    
    # Train agent
    results = train_policy_gradient(
        env_name=args.env,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        use_reward_to_go=args.reward_to_go,
        use_advantage_normalization=args.advantage_norm,
        use_baseline=args.baseline,
        hidden_sizes=args.hidden_sizes,
        max_episode_length=args.max_episode_length,
        print_freq=args.print_freq,
        save_path=args.save_path,
        seed=args.seed
    )
    
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Final mean return: {results['iteration_returns'][-1]:.2f}")
    print(f"Best mean return: {max(results['iteration_returns']):.2f}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
