import gym
import numpy as np
import collections

class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super.__init__(RepeatActionAndMaxFrame, self)
        self.env = env
        self.frame_buffer = np.zeros(env.observation_space * 2)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0
        done = False
        max_frame = 0

        for i in range(self.repeat):
            observation, reward, done, info = self.env.step(action)
            self.frame_buffer[i] = observation
            total_reward += reward

            if done:
                break

        return max_frame, total_reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.frame_buffer[0] = observation

        return observation

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, new_shape):
        super.__init__(PreprocessFrame, self)
        self.new_shape = new_shape
        self.env = env

    def observation(self, observation):
        # convert obs to grey scale
        obs = np.dot(observation, [0.2989, 0.5870, 0.1140])
        # resize the image to the new size
        obs = np.resize(obs, self.new_shape)
        # move observation channel axis to position 0 from 2
        obs = np.moveaxis(obs, 2, 0)
        # divide by 255
        obs /= 255
        return obs

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, stack_size):
        super.__init__(StackFrames, self)
        self.env = env
        self.stack = collections.deque(maxlen=stack_size)


    def reset(self):
        # clear the stack
        self.stack.clear()
        
        # reset the env
        obs = self.env.reset()

        for i in range(stack_size):
            self.stack.append(obs)
        stack = np.asarray(list(self.stack))
        stack = np.reshape(stack, self.env.observation_space.low)
        return stack

    def observation(self, observation):
        self.stack.append(observation)
        stack = np.asarray(list(self.stack))
        stack = np.reshape(stack, self.env.observation_space.low)
        return stack


def make_env(env_name, new_shape, stack_size):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env)
    env = PreprocessFrame(env, new_shape)
    env = StackFrames(env, stack_size)

    return env





