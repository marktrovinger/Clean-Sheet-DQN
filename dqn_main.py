import gym
from dqn_agent import DQN_Agent
import numpy as np
import spinup

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    n_games = 10000
    no_op = 30

    agent = DQN_Agent(gamma=0.99, input_size=env.input_size, epsilon=1, n_actions=env.n_actions, memory_size=10000, fc_input_dims=512,lr=0.000025)

    observation = env.reset()
    # TODO: using our implementation, run the number of games specified
    for _ in range(n_games):
        action = env.r

    # TODO: using the reference spinningup algo, run the same number of games
    #       minor problem, apparently SpinningUp doesn't have DQN implementation,
    #       so that plan won't work
    # TODO: investigate the number of games that will give good results for cartpole