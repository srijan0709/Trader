import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1440)
        self.fc2 = nn.Linear(1440,1024)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256,128)
        self.fc6 = nn.Linear(128,64)
        self.fc7 = nn.Linear(64,32)
        self.fc8 = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = self.fc8(x)
        return x
    


class DQNAgent:
    def __init__(self, env, policy_net, target_net, batch_size=4096, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.policy_net = policy_net
        self.target_net = target_net
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(policy_net.parameters())
        self.loss_fn = nn.MSELoss()
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample_memory(self):
        return random.sample(self.memory, self.batch_size)
    
    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.sample_memory()
        batch = list(zip(*transitions))
        states =np.array(batch[0])

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor([float(r) for r in batch[2]], dtype=torch.float32).unsqueeze(1).to(device)

        next_states = np.array([np.array(s, dtype=np.float32) for s in batch[3]])
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(device)

        
        current_q_values = self.policy_net(states).gather(1, actions)
        max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        loss = self.loss_fn(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)