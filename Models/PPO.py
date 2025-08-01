import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Tuple, List
import torch.nn.functional as F
from torch.distributions import Categorical
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

class PPOActor(nn.Module):
    """Actor network for PPO - outputs action probabilities"""
    def __init__(self, input_dim, output_dim):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = F.softmax(self.fc5(x), dim=-1)
        return x

class PPOCritic(nn.Module):
    """Critic network for PPO - outputs state value"""
    def __init__(self, input_dim):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class PPOMemory:
    """Memory buffer for PPO training"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def store(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        
    def get_batch(self):
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(device)
        actions = torch.tensor(self.actions, dtype=torch.long).to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        values = torch.tensor(self.values, dtype=torch.float32).to(device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device)
        
        return states, actions, rewards, log_probs, values, dones

class PPOAgent:
    def __init__(self, env, input_dim, output_dim, lr=3e-4, gamma=0.99, clip_ratio=0.2, 
                 value_coef=0.5, entropy_coef=0.01, update_epochs=10, batch_size=64):
        """
        PPO Agent
        
        Args:
            env: Trading environment
            input_dim: Input dimension (number of features)
            output_dim: Output dimension (number of actions)
            lr: Learning rate
            gamma: Discount factor
            clip_ratio: PPO clipping ratio
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            update_epochs: Number of epochs to update policy
            batch_size: Batch size for training
        """
        self.env = env
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # Initialize networks
        self.actor = PPOActor(input_dim, output_dim).to(device)
        self.critic = PPOCritic(input_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Memory
        self.memory = PPOMemory()
        
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs = self.actor(state)
            value = self.critic(state)
            
        # Sample action from probability distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in memory"""
        self.memory.store(state, action, reward, log_prob, value, done)
    
    def compute_returns_and_advantages(self, rewards, values, dones):
        """Compute returns and advantages using GAE"""
        returns = []
        advantages = []
        gae = 0
        
        # Convert to numpy for easier computation
        rewards = rewards.cpu().numpy()
        values = values.cpu().numpy()
        dones = dones.cpu().numpy()
        
        # Add a dummy value for the last state
        values = np.append(values, 0)
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
                
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae  # 0.95 is lambda for GAE
            advantages.append(gae)
            returns.append(gae + values[i])
        
        advantages.reverse()
        returns.reverse()
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update_policy(self):
        """Update policy using PPO"""
        if len(self.memory.states) == 0:
            return
        
        # Get batch from memory
        states, actions, rewards, old_log_probs, values, dones = self.memory.get_batch()
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(rewards, values, dones)
        
        # Update policy for multiple epochs
        for _ in range(self.update_epochs):
            # Forward pass
            action_probs = self.actor(states)
            new_values = self.critic(states).squeeze()
            
            # Calculate new log probabilities
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Calculate ratio for PPO
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(new_values, returns)
            
            # Total loss
            total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        # Clear memory after update
        self.memory.clear()
    
    def save_models(self, actor_path, critic_path):
        """Save both actor and critic models"""
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    def load_models(self, actor_path, critic_path):
        """Load both actor and critic models"""
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))