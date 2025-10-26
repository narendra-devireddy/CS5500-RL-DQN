"""
Policy Gradient Agent with Variance Reduction Techniques
Implements REINFORCE algorithm with:
- Reward-to-go (reduces variance)
- Advantage normalization with baseline
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    """Neural network for policy approximation"""
    
    def __init__(self, state_dim, action_dim, hidden_sizes=[64, 64]):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.Tanh())
            input_dim = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass to get action logits"""
        return self.network(state)
    
    def get_action_probs(self, state):
        """Get action probabilities"""
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """Neural network for value function (baseline)"""
    
    def __init__(self, state_dim, hidden_sizes=[64, 64]):
        super(ValueNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.Tanh())
            input_dim = hidden_size
        
        # Output layer (single value)
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass to get state value"""
        return self.network(state).squeeze(-1)


class PolicyGradientAgent:
    """
    Policy Gradient Agent with variance reduction techniques
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        use_reward_to_go=True,
        use_advantage_normalization=True,
        use_baseline=True,
        hidden_sizes=[64, 64]
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate
            gamma: Discount factor
            use_reward_to_go: If True, use reward-to-go instead of total trajectory reward
            use_advantage_normalization: If True, normalize advantages
            use_baseline: If True, use value function baseline
            hidden_sizes: Hidden layer sizes for networks
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.use_advantage_normalization = use_advantage_normalization
        self.use_baseline = use_baseline
        
        # Policy network
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Value network (baseline)
        if self.use_baseline:
            self.value_net = ValueNetwork(state_dim, hidden_sizes).to(device)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Storage for trajectories
        self.reset_trajectories()
    
    def reset_trajectories(self):
        """Reset trajectory storage"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
    
    def select_action(self, state, training=True):
        """
        Select action using current policy
        
        Args:
            state: Current state
            training: If True, store trajectory information
        
        Returns:
            action: Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs = self.policy_net.get_action_probs(state_tensor)
        
        # Sample action from probability distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        
        if training:
            # Store log probability for policy gradient
            log_prob = dist.log_prob(action)
            self.log_probs.append(log_prob)
            
            # Store state and action
            self.states.append(state)
            self.actions.append(action.item())
            
            # Store value estimate if using baseline
            if self.use_baseline:
                with torch.no_grad():
                    value = self.value_net(state_tensor)
                self.values.append(value.item())
        
        return action.item()
    
    def store_reward(self, reward):
        """Store reward for current step"""
        self.rewards.append(reward)
    
    def compute_returns(self):
        """
        Compute returns for each timestep
        
        Returns:
            returns: List of returns (either total reward or reward-to-go)
        """
        T = len(self.rewards)
        returns = []
        
        if self.use_reward_to_go:
            # Reward-to-go: G_t = sum_{t'=t}^{T} gamma^{t'-t} * r_{t'+1}
            G = 0
            for t in reversed(range(T)):
                G = self.rewards[t] + self.gamma * G
                returns.insert(0, G)
        else:
            # Total trajectory reward: G = sum_{t=0}^{T} gamma^t * r_{t+1}
            G = 0
            for t in range(T):
                G += (self.gamma ** t) * self.rewards[t]
            returns = [G] * T
        
        return returns
    
    def compute_advantages(self, returns):
        """
        Compute advantages using baseline
        
        Args:
            returns: List of returns
        
        Returns:
            advantages: List of advantages
        """
        if self.use_baseline:
            # Advantage = Return - Baseline
            advantages = [ret - val for ret, val in zip(returns, self.values)]
        else:
            # No baseline, advantages are just returns
            advantages = returns
        
        # Normalize advantages
        if self.use_advantage_normalization and len(advantages) > 1:
            advantages = np.array(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.tolist()
        
        return advantages
    
    def update(self):
        """
        Update policy using collected trajectories
        
        Returns:
            policy_loss: Policy loss value
            value_loss: Value loss value (if using baseline)
        """
        if len(self.rewards) == 0:
            return None, None
        
        # Compute returns
        returns = self.compute_returns()
        
        # Compute advantages
        advantages = self.compute_advantages(returns)
        
        # Convert to tensors
        log_probs = torch.stack(self.log_probs).to(device)
        advantages_tensor = torch.FloatTensor(advantages).to(device)
        returns_tensor = torch.FloatTensor(returns).to(device)
        
        # Policy gradient loss: -E[log π(a|s) * A(s,a)]
        policy_loss = -(log_probs * advantages_tensor).mean()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value network (if using baseline)
        value_loss = None
        if self.use_baseline:
            states_tensor = torch.FloatTensor(np.array(self.states)).to(device)
            predicted_values = self.value_net(states_tensor)
            
            # MSE loss between predicted values and returns
            value_loss = F.mse_loss(predicted_values, returns_tensor)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            value_loss = value_loss.item()
        
        return policy_loss.item(), value_loss
    
    def save(self, filepath):
        """Save model"""
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
        }
        if self.use_baseline:
            checkpoint['value_net'] = self.value_net.state_dict()
            checkpoint['value_optimizer'] = self.value_optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        
        if self.use_baseline and 'value_net' in checkpoint:
            self.value_net.load_state_dict(checkpoint['value_net'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
