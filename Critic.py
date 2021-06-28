import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Critic(nn.Module):
    """ Critic model for DDPG """

    def __init__(self, state_size, action_size, seed=0):
        """Constructor for QNetwork model to initialize states, actions and random seed
        Args:
            state_size:  number of states
            action_size: number of actions
            seed: rng seed value
        """

        super(Critic, self).__init__()
        fc1_units = 400
        fc2_units = 300
        fc3_units = 300
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(fc1_units)
        self.fcs1 = nn.Linear(state_size, fc1_units)       # First Layer
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)  # Second Layer
        self.fc3 = nn.Linear(fc2_units, fc3_units)                # Third Layer
        self.fc4 = nn.Linear(fc3_units, 1)


        # fc1 - calculate uniform distribution of -1/Sqrt(fan-in fc1)
        #fc1_fin = self.fc1.weight.data.size()[0]
        #ud_fc1 = 1.0 / np.sqrt(fc1_fin)
        #self.fc1.weight.data.uniform_(*(-ud_fc1, ud_fc1))

        # fc2 - calcuate uniform distribution of -1/sqrt(fan_int fc2)
        #fc2_fin = self.fc2.weight.data.size()[0]
        #ud_fc2 = 1.0 / np.sqrt(fc2_fin)
        #self.fc2.weight.data.uniform_(*(-ud_fc2, ud_fc2))

        # fc3 - uniform distribution of fc3
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Network of state to action values
        Args:
            state: state to map to an action
        Returns:
            mapped state to action values
        """
        xs = F.relu(self.bn(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)