from dqn_network import DeepQNetwork
from replay_buffer import ReplayBuffer

class DQN_Agent():
    def __init__(self, gamma, input_size, epsilon, n_actions, memory_size, fc_input_dims, lr, no_op=30):
        self.replay_buffer = ReplayBuffer(memory_size)
        self.no_op = no_op

        self.q = DeepQNetwork(gamma, input_size, epsilon, n_actions, fc_input_dims, lr)

        # TODO: finish implementing init function

    # TODO: implement the learning algorithmS
    def learn(self):
        pass