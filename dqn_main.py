import gym
from dqn_agent import DQN_Agent
import numpy as np
import spinup

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    n_games = 10000


    # TODO: using our implementation, run the number of games specified
    for _ in range(n_games):
        pass

    # TODO: using the reference spinningup algo, run the same number of games
    #       minor problem, apparently SpinningUp doesn't have DQN implementation,
    #       so that plan won't work
    # TODO: investigate the number of games that will give good results for cartpole