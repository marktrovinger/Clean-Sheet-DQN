import numpy as np

class ReplayBuffer():
    def __init__(self, memory_size):
        state = np.zeroes(memory_size)
    
    def store_transition(self, state, action, rewards, state_, done)
        pass

    def sample_memory(self, batch_size):
        pass