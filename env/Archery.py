import numpy as np
import gym
from math import pi, cos, sin
import itertools


def wind_dir_trans(state):
    # wind velocity transition probability
    dv_x = np.random.uniform(low=-2, high=2)
    dv_y = np.random.uniform(low=-2, high=2)

    next_wind= state[0:2,] + [dv_x,dv_y]

    return next_wind


def polar2cartesian(v):
    r = int(v[0])
    theta = v[1]
    return [r*cos(theta), r*sin(theta)]


def wind_coeff(humidity):
    return-0.005*humidity+1


def get_score(state, action):
    eff_wind_vector = wind_coeff(state[2]) * state[0:2]  # effective wind vector
    aiming_vector = polar2cartesian(action)
    reach_point = np.array(eff_wind_vector-aiming_vector)
    score = 11 - np.linalg.norm(reach_point, ord=2)
    score = int(score)
    score = score if score > 0 else 0
    return score


class ArcheryEnv:
    def __init__(self, distance=50, v0=1, vw_max=10, step_max=50):
        # Define the state space
        self.velocity_space = gym.spaces.Box(low=-vw_max, high=vw_max, shape=(2,), dtype=np.float32)
        self.humidity_set = np.array([0, 25, 50, 75, 100])
        self.humidity_space = gym.spaces.Discrete(self.humidity_set.shape[0])

        self.state_dim = self.velocity_space.shape[0] + 1

        # Define the action set
        # k = 1  # have to modify
        # r_space = np.array([k, 2*k, 3*k,4*k, 5*k])
        r_space = np.array([i for i in range(1, vw_max+1)])
        theta_space = np.array([i/24*2*pi for i in range(0, 24)])

        self.action_set = np.array([i for i in itertools.product(r_space, theta_space)])
        self.action_num = self.action_set.shape[0]
        self.action_space = gym.spaces.Discrete(self.action_num)

        # Number of step per each episode
        self.step_max = 100
        self.time_step = 0

        # Initial state setting
        self.state = [0, 0, 0]
        self.reward = 0
        self.cum_score = 0

    def step(self, action):
        action = self.action_set[action]
        next_state = wind_dir_trans(self.state)
        next_state = np.clip(next_state,self.velocity_space.low, self.velocity_space.high)
        next_state = np.append(next_state, self.state[2])

        self.time_step += 1
        done = self.time_step >= self.step_max

        score = get_score(self.state, action)

        self.cum_score = self.cum_score + score
        if score > 5:
            self.reward = score ^ 2
        else:
            self.reward = score ^ 2 - 50

        return np.array(next_state), self.reward, done, {}

    def reset(self):
        self.time_step = 0
        h = self.humidity_set[np.random.randint(0, self.humidity_set.shape[0] - 1)]
        w = np.random.uniform(low=self.velocity_space.low, high=self.velocity_space.high)
        self.state = np.append(w,h)
        self.reward = 0
        self.cum_score = 0
        return np.array(self.state)