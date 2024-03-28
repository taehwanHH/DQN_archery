import numpy as np
import gym
from math import pi, cos, sin
import itertools
from param import Hyper_Param
from Archery_env.env_func import *


class ArcheryEnv:
    def __init__(self, vw_max=Hyper_Param['vw_max'], step_max=Hyper_Param['step_max']):
        # Define the state space
        self.v_r_space = gym.spaces.Box(low=0, high=vw_max, shape=(1,), dtype=np.float32)
        self.v_theta_space = gym.spaces.Box(low=0, high=2*pi, shape=(1,), dtype=np.float32)
        self.humidity_set = np.array([0, 25, 50, 75, 100])
        self.humidity_space = gym.spaces.Discrete(self.humidity_set.shape[0])

        self.state_dim = self.v_r_space.shape[0] + self.v_theta_space.shape[0] + 1

        self.vw_max = vw_max

        # Define the action set
        r_space = np.array([0.5*i for i in range(1, 2*vw_max+1)])
        theta_space = np.array([i/45*2*pi for i in range(0, 45)])

        self.action_set = np.array([i for i in itertools.product(r_space, theta_space)])
        self.action_set = np.append(self.action_set, np.array([[0,0]]), axis=0)

        # self.cartesian_action_set = v_polar2cartesian(np.array(self.action_set))
        self.action_num = self.action_set.shape[0]
        self.action_space = gym.spaces.Discrete(self.action_num)

        # Number of step per each episode
        self.step_max = step_max
        self.time_step = 0

        # Initial state setting
        self.state = [0, 0, 0]
        self.reward = 0
        self.cum_score = 0
        # self.cum_rand_score = 0
        # self.cum_optimal_score = 0

    def step(self, action):
        action = self.action_set[action]

        # # Random action select
        # rand_idx = np.random.randint(0, self.action_num)
        # rand_action = self.action_set[rand_idx]
        #
        # [_, rand_score] = get_score(self.state, rand_action)
        # self.cum_rand_score += rand_score
        #
        # # Optimal policy
        # optimal_action = optimal_policy(self.state, self.cartesian_action_set, self.action_set)
        # [_, optimal_score] = get_score(self.state, optimal_action)
        # self.cum_optimal_score += optimal_score

        next_wind = wind_dir_trans(self.state)
        wind_mag = np.linalg.norm(np.array(next_wind), ord=2)

        if wind_mag > self.vw_max:
            next_wind = (self.vw_max / wind_mag) * next_wind

        next_wind = cartesian2polar(next_wind)
        # next_state = np.append(next_wind, self.state[2])
        next_humidity = humidity_trans(self.state, self.humidity_set)
        next_state = np.append(next_wind, next_humidity)

        self.time_step += 1
        done = self.time_step >= self.step_max

        [distance, score] = get_score(self.state, action)

        self.cum_score += score

        # reward setting
        self.reward = (-distance+score)*6

        return np.array(next_state), self.reward, done, {}

    def reset(self):
        self.time_step = 0
        h = self.humidity_set[np.random.randint(0, self.humidity_set.shape[0])]
        r = np.random.uniform(low=0, high=self.vw_max, size=(1,))
        theta = np.random.uniform(low=0, high=2*pi, size=(1,))
        w = [r,theta]
        self.state = np.append(w, h)
        self.reward = 0
        self.cum_score = 0
        # self.cum_rand_score = 0
        # self.cum_optimal_score = 0

        return np.array(self.state)
