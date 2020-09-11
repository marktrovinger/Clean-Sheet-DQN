from torch import nn 
import torch.functional as F
from replay_buffer import ReplayBuffer

class DeepQNetwork(nn.Module):
    def __init__(self, gamma, input_size, epsilon, n_actions):
        super.__init__(DeepQNetwork, self)
        self.gamma = gamma
        self.epsilon = epsilon

        #self.replay_buffer = ReplayBuffer(memory_size)
        
        self.input_layer = nn.Linear(input_size, 32)
        self.conv1 = nn.Conv2d(32, 64, )