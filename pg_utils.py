"""
Utility functions for Policy Gradient training and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_learning_curves(results_dict, env_name, window=10, save_path=None):
    """
    Plot learning curves comparing different configurations
    
    Args:
        results_dict: Dictionary mapping config names to results
        env_name: Environment name
        window: Window size for moving average
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot 1: Average returns
    ax = axes[0]
    for idx, (config_name, results) in enumerate(results_dict.items()):
        returns = results['iteration_returns']
        iterations = range(1, len(returns) + 1)
        
        color = colors[idx % len(colors)]
        
        # Plot raw data with transparency
        ax.plot(iterations, returns, alpha=0.3, color=color)
        
        # Plot moving average
        if len(returns) >= window:
            smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
            smooth_x = range(window, len(returns) + 1)
            ax.plot(smooth_x, smoothed, label=config_name, linewidth=2, color=color)
        else:
            ax.plot(iterations, returns, label=config_name, linewidth=2, color=color)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Average Return', fontsize=12)
    ax.set_title(f'{env_name} - Learning Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Average episode lengths
    ax = axes[1]
    for idx, (config_name, results) in enumerate(results_dict.items()):
        lengths = results['iteration_lengths']
        iterations = range(1, len(lengths) + 1)
        
        color = colors[idx % len(colors)]
        
        # Plot raw data with transparency
        ax.plot(iterations, lengths, alpha=0.3, color=color)
        
        # Plot moving average
        if len(lengths) >= window:
            smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
            smooth_x = range(window, len(lengths) + 1)
            ax.plot(smooth_x, smoothed, label=config_name, linewidth=2, color=color)
        else:
            ax.plot(iterations, lengths, label=config_name, linewidth=2, color=color)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Average Episode Length', fontsize=12)
    ax.set_title(f'{env_name} - Episode Lengths', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_single_training_curve(results, env_name, window=10, save_path=None):
    """
    Plot training curves for a single configuration
    
    Args:
        results: Training results dictionary
        env_name: Environment name
        window: Window size for moving average
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Average returns
    ax = axes[0, 0]
    returns = results['iteration_returns']
    iterations = range(1, len(returns) + 1)
    
    ax.plot(iterations, returns, alpha=0.3, label='Raw', color='blue')
    
    if len(returns) >= window:
        smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
        smooth_x = range(window, len(returns) + 1)
        ax.plot(smooth_x, smoothed, label=f'{window}-iter MA', linewidth=2, color='blue')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Return')
    ax.set_title('Average Return per Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Average episode lengths
    ax = axes[0, 1]
    lengths = results['iteration_lengths']
    
    ax.plot(iterations, lengths, alpha=0.3, label='Raw', color='green')
    
    if len(lengths) >= window:
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        smooth_x = range(window, len(lengths) + 1)
        ax.plot(smooth_x, smoothed, label=f'{window}-iter MA', linewidth=2, color='green')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Episode Length')
    ax.set_title('Average Episode Length per Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Policy loss
    ax = axes[1, 0]
    if results['policy_losses']:
        losses = results['policy_losses']
        ax.plot(losses, alpha=0.5, label='Policy Loss', color='red')
        
        if len(losses) >= window:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            smooth_x = range(window-1, len(losses))
            ax.plot(smooth_x, smoothed, label=f'{window}-step MA', linewidth=2, color='red')
        
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Value loss (if available)
    ax = axes[1, 1]
    if results['value_losses'] and any(v is not None for v in results['value_losses']):
        losses = [v for v in results['value_losses'] if v is not None]
        ax.plot(losses, alpha=0.5, label='Value Loss', color='purple')
        
        if len(losses) >= window:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            smooth_x = range(window-1, len(losses))
            ax.plot(smooth_x, smoothed, label=f'{window}-step MA', linewidth=2, color='purple')
        
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss (Baseline)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{env_name} - Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def compare_configurations(results_dict, env_name, save_path=None):
    """
    Create comparison table and bar chart for different configurations
    
    Args:
        results_dict: Dictionary mapping config names to results
        env_name: Environment name
        save_path: Path to save plot
    """
    # Extract final performance metrics
    config_names = []
    final_returns = []
    best_returns = []
    final_lengths = []
    
    for config_name, results in results_dict.items():
        config_names.append(config_name)
        
        returns = results['iteration_returns']
        final_returns.append(np.mean(returns[-10:]))  # Last 10 iterations
        best_returns.append(np.max(returns))
        
        lengths = results['iteration_lengths']
        final_lengths.append(np.mean(lengths[-10:]))
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x_pos = np.arange(len(config_names))
    width = 0.35
    
    # Plot 1: Returns comparison
    ax = axes[0]
    bars1 = ax.bar(x_pos - width/2, final_returns, width, label='Final (last 10)', 
                   alpha=0.8, color='skyblue', edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, best_returns, width, label='Best', 
                   alpha=0.8, color='lightcoral', edgecolor='black')
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Average Return', fontsize=12)
    ax.set_title(f'{env_name} - Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Episode lengths
    ax = axes[1]
    bars = ax.bar(x_pos, final_lengths, alpha=0.8, color='lightgreen', edgecolor='black')
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Average Episode Length', fontsize=12)
    ax.set_title(f'{env_name} - Episode Length Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()
    
    # Print comparison table
    print(f"\n{'='*80}")
    print(f"Performance Comparison - {env_name}")
    print(f"{'='*80}")
    print(f"{'Configuration':<30} {'Final Return':<15} {'Best Return':<15} {'Final Length':<15}")
    print(f"{'-'*80}")
    for i, config_name in enumerate(config_names):
        print(f"{config_name:<30} {final_returns[i]:<15.2f} {best_returns[i]:<15.2f} {final_lengths[i]:<15.2f}")
    print(f"{'='*80}\n")


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
