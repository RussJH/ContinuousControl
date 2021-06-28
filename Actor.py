import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """ Actor model for DDPG """

    def __init__(self, state_size, action_size, seed=0):
        """Constructor for QNetwork model to initialize states, actions and random seed
        Args:
            state_size:  number of states
            action_size: number of actions
            seed: rng seed value
        """

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 400)   # First Layer
        self.fc2 = nn.Linear(400, 300)          # Second Layer
        self.fc3 = nn.Linear(300, action_size)  # Third Layer

        # fc1 - calculate uniform distribution of -1/Sqrt(fan-in fc1)
        #fc1_fin = self.fc1.weight.data.size()[0]
        #ud_fc1 = 1.0 / np.sqrt(fc1_fin)
        #self.fc1.weight.data.uniform_(*(-ud_fc1, ud_fc1))

        # fc2 - calcuate uniform distribution of -1/sqrt(fan_int fc2)
        #fc2_fin = self.fc2.weight.data.size()[0]
        #ud_fc2 = 1.0 / np.sqrt(fc2_fin)
        #self.fc2.weight.data.uniform_(*(-ud_fc2, ud_fc2))

        # fc3 - uniform distribution of fc3
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Network of state to action values
        Args:
            state: state to map to an action
        Returns:
            mapped state to action values
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))