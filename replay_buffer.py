import numpy as np

class ReplayBuffer():
    def __init__(self, memory_size):
        self.state = np.zeroes((memory_size), dtype=np.float64)
        self.action = np.zeros((memory_size), dtype=np.float64)
        self.rewards = np.zeros((memory_size), dtype=np.float64)
        self.state_ = np.zeros((memory_size), dtype=np.float64)
        self.done = np.zeros((memory_size), dtype=np.int8)
        self.mem_cntr = 0

    
    def store_transition(self, state, action, rewards, state_, done):
        self.state[self.mem_cntr] = state
        self.action[self.mem_cntr] = action
        self.rewards[self.mem_cntr] = rewards
        self.state_[self.mem_cntr] = state_
        self.done[self.mem_cntr] = done
        self.mem_cntr += 1

    def sample_memory(self, batch_size):
        if self.mem_cntr < batch_size:
            return
        
        state = self.state[batch_size]
        action = self.action[batch_size]
        rewards = self.rewards[batch_size]
        state_ = self.state_[batch_size]
        done = self.done[batch_size]
        
        return state, action, rewards, state_, done