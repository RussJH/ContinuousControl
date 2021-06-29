# Continuous Control Project Report

For instructions to run see [README.md](./README.md)

## Implementation Details

The Project consists of the following files:

    - Agent.py                  -- Implementation of a Deep Deterministic Policy Gradient (DDPG) Agent
    - Actor.py                  -- Actor (Policy) model used by the agent
    - Critic.py                 -- Critic (Value) model used by the agent 
    - ReplayBuffer.py           -- Double ended Queue with random sampling methods to store the learning experniences
    - ContinuousControl.ipynb   -- Jupyter notebook used to train the DDPG Agent 


### Learning Algorithm and Neural Network

TDOD

### Agent Hyper Parameters
|Parameter| Value|
--- | --- |
Epsilon start | 1.0 |
Epsilon decay | 0.995 |
epsilon minimum | 0.01 |
Discount factor | 0.99 |
Soft update rate | 1e-3 |
Learing Rate | 5e-4 |

### Agent Hyper Parameters
|Parameter| Value|
--- | --- |

### Actor (Policy) Hyper Parameters
|Parameter| Value|
--- | --- |

### Critic (Value) Hyper Parameters
|Parameter| Value|
--- | --- |

## Buffer Hyper Parameters
|Parameter| Value|
--- | --- |
Replay buffer size | 1e5 |
Batch size | 64 |

## Plot
 The figure below is the plot of the rewards over runs during the training episodes

<img src="Images/ContinuousControl.png"/>

 > [Episode 100]	Average Score (Last 100 episodes): 2.0
 > [Episode 200]	Average Score (Last 100 episodes): 7.5
 > [Episode 300]	Average Score (Last 100 episodes): 11.4
 > [Episode 400]	Average Score (Last 100 episodes): 17.5
 > [Episode 500]	Average Score (Last 100 episodes): 22.0
 > [Episode 600]	Average Score (Last 100 episodes): 24.5
 > [Episode 700]	Average Score (Last 100 episodes): 24.9
 > [Episode 776]	Average Score (Last 100 episodes): 30.1
 > Solved in 775 episodes!	Average Score (Last 100 episodes)=30.1
 
## Future work

TODO 