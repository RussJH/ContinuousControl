
from Actor import Actor
from Critic import Critic
from Noise import Noise
from ReplayBuffer import ReplayBuffer

import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Deep Q-Network Agent """

    def __init__(self, state_size, action_size, seed=0):
        """Initialize Agent
        Args:
            state_size: number of possible states
            action_size: number of possible acitons
            seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1
        self.discount_factor = 1e-6
        self.t_step = 0

        # Actor model
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=1e-4) # Learning rate

        # Critic Model
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=1e-3, weight_decay=0) # Learning rate

        # Add noise
        self.noise = Noise(action_size, seed)

        # Memory buffer
        self.memory = ReplayBuffer(action_size, seed=seed)

        # Make sure target is initialized with the same weight as the source
        #self.hard_update(self.actor_target, self.actor_local)
        #self.hard_update(self.critic_target, self.critic_local)
    
    def reset_noise(self):
        """ Reset noise"""
        self.noise.reset()

    def step(self, state, action, reward, next_state, done, step):
        """Save the experience in the buffer
        Args:
            state: current state
            action: selected action
            reward: reward given
            next_state: next state to advance to
            done: has the memory completed
        """
        # save experience 
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.memory.batch_size and step % 4 == 0:
            experiences = self.memory.get_random_sample()
            self.learn(experiences, 0.99) # discount factor

    def learn(self, experiences, gamma):
        """Teachs the agent based on the current state, action reward and next state
        
        Args:
            experiences: tuple of state, actions, rewards, next states and done
            gamma: the discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Update critic network
        # Get next state-action and Q values
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor network
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Updates networks
        self.update(self.critic_local, self.critic_target, 1e-3)
        self.update(self.actor_local, self.actor_target, 1e-3)

        self.epsilon -= self.discount_factor
        self.noise.reset()

    def update(self, local_model, target_model, tau):
        """Update the model params
        target = t*local+(1-t)*target

        Args:
            local_model: pytorch model source
            target_model: pytorch model destination
            tau: interpolation param
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    ## initialze the target network same as the source network
    def hard_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)
    
    def get_action(self, state, add_noise=True):
        """Gets the action from the given state
        Args:
            state: current state
            eps: epsilon value for a epsilon-greedy policy
        Returns:
            action selection based on the epsilon-greedy policy
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon * self.noise.sample()
        return np.clip(action, -1, 1)