import torch as T
from torch import nn 
import torch.nn.functional as F
import torch.optim as optim
from replay_buffer import ReplayBuffer

class DeepQNetwork(nn.Module):
    def __init__(self, gamma, input_size, epsilon, n_actions, memory_size, fc_input_dims, lr):
        super.__init__(DeepQNetwork, self)
        self.gamma = gamma
        self.epsilon = epsilon

        self.replay_buffer = ReplayBuffer(memory_size)
        
        #self.input_layer = nn.Linear(input_size, 32)
        self.conv1 = nn.Conv2d(input_size[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        self.fc = nn.Linear(fc_input_dims, 512)
        self.output_layer = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self, state):
        x = self.conv2(F.relu(self.conv1(state)))
        x = self.conv3(F.relu(self.conv2(x)))