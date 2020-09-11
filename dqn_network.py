from torch import nn 
import torch.functional as F
from replay_buffer import ReplayBuffer

class DeepQNetwork(nn.Module):
    def __init__(self, gamma):
        super.__init__(DeepQNetwork)