"""
Utility functions for training and visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_training_curve(results, env_name, window=100, save_path=None):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(results['episode_rewards'], alpha=0.3, label='Episode Reward')
    if len(results['episode_rewards']) >= window:
        moving_avg = np.convolve(results['episode_rewards'], 
                                 np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(results['episode_rewards'])), 
                       moving_avg, label=f'{window}-Episode Moving Average', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title(f'{env_name} - Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean rewards over time
    if results['mean_rewards']:
        axes[0, 1].plot(results['mean_rewards'], label='Mean 100-Episode Reward', linewidth=2)
        axes[0, 1].axhline(y=results['best_mean_reward'], color='r', 
                          linestyle='--', label=f'Best Mean: {results["best_mean_reward"]:.2f}')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Mean Reward')
        axes[0, 1].set_title(f'{env_name} - Mean Reward (100 episodes)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[1, 0].plot(results['episode_lengths'], alpha=0.3, label='Episode Length')
    if len(results['episode_lengths']) >= window:
        moving_avg = np.convolve(results['episode_lengths'], 
                                 np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window-1, len(results['episode_lengths'])), 
                       moving_avg, label=f'{window}-Episode Moving Average', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].set_title(f'{env_name} - Episode Lengths')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training loss
    if results['losses']:
        axes[1, 1].plot(results['losses'], alpha=0.5, label='Training Loss')
        if len(results['losses']) >= window:
            moving_avg = np.convolve(results['losses'], 
                                     np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(results['losses'])), 
                           moving_avg, label=f'{window}-Episode Moving Average', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title(f'{env_name} - Training Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_mountaincar_policy(agent, resolution=50):
    """
    Plot the action choices of trained agent for MountainCar
    across different position and velocity values
    """
    # Define ranges for position and velocity
    position_range = np.linspace(-1.2, 0.6, resolution)
    velocity_range = np.linspace(-0.07, 0.07, resolution)
    
    # Create meshgrid
    pos_grid, vel_grid = np.meshgrid(position_range, velocity_range)
    
    # Get action for each state
    action_grid = np.zeros_like(pos_grid)
    q_value_grid = np.zeros_like(pos_grid)
    
    for i in range(resolution):
        for j in range(resolution):
            state = np.array([pos_grid[i, j], vel_grid[i, j]])
            q_values = agent.get_q_values(state)
            action_grid[i, j] = np.argmax(q_values)
            q_value_grid[i, j] = np.max(q_values)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Action map
    im1 = axes[0].contourf(pos_grid, vel_grid, action_grid, levels=[-0.5, 0.5, 1.5, 2.5], 
                           colors=['blue', 'gray', 'red'], alpha=0.6)
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Velocity')
    axes[0].set_title('MountainCar - Action Policy\n(Blue=Left, Gray=No Push, Red=Right)')
    axes[0].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0], ticks=[0, 1, 2])
    cbar1.set_label('Action')
    cbar1.set_ticklabels(['Left (0)', 'No Push (1)', 'Right (2)'])
    
    # Q-value map
    im2 = axes[1].contourf(pos_grid, vel_grid, q_value_grid, levels=20, cmap='viridis')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Velocity')
    axes[1].set_title('MountainCar - Max Q-Values')
    axes[1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Max Q-Value')
    
    plt.tight_layout()
    plt.savefig('mountaincar_policy.png', dpi=300, bbox_inches='tight')
    print("Policy plot saved to mountaincar_policy.png")
    plt.show()


def save_results(results, filepath):
    """Save training results to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filepath}")


def load_results(filepath):
    """Load training results from file"""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    print(f"Results loaded from {filepath}")
    return results
