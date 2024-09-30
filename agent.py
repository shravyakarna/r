#!/usr/bin/env python

import random
import pandas as pd
import numpy as np


class AgentBasic(object):
    def __init__(self):
        pass

    def act(self, obs):
        angle = obs[2]
        return 0 if angle < 0 else 1


class AgentRandom(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()


class AgentLearning(object):
    def __init__(self, env, alpha=0.1, epsilon=1.0, gamma=0.9):
        self.env = env
        self.alpha = alpha          
        self.epsilon = epsilon
        self.gamma = gamma          
        self.Q_table = dict()
        self._set_seed()
        self.training_trials = 0
        self.testing_trials = 0

    def _set_seed(self):
        np.random.seed(21)
        random.seed(21)

    def build_state(self, features):
        return int("".join(map(lambda feature: str(int(feature)), features)))

    def create_state(self, obs):
        cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
        pole_angle_bins = pd.cut([-2, 2], bins=10, retbins=True)[1][1:-1]
        cart_velocity_bins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
        angle_rate_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]
        state = self.build_state([np.digitize(x=[obs[0]], bins=cart_position_bins)[0],
                                 np.digitize(x=[obs[1]], bins=pole_angle_bins)[0],
                                 np.digitize(x=[obs[2]], bins=cart_velocity_bins)[0],
                                 np.digitize(x=[obs[3]], bins=angle_rate_bins)[0]])
        return state

    def choose_action(self, state):
        
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            max_Q = self.get_maxQ(state)
        actions = []
        for key, value in self.Q_table[state].items():
            if value == max_Q:
                actions.append(key)
            if len(actions) != 0:
                action = random.choice(actions)
        return action

    def create_Q(self, state, valid_actions):
        
        if state not in self.Q_table:
            self.Q_table[state] = dict()
            for action in valid_actions:
                self.Q_table[state][action] = 0.0
        return

    def get_maxQ(self, state):
        
        maxQ = max(self.Q_table[state].values())
        return maxQ

    def learn(self, state, action, prev_reward, prev_state, prev_action):
    
        self.Q_table[prev_state][prev_action] = (1 - self.alpha) * \
            self.Q_table[prev_state][prev_action] + self.alpha * \
            (prev_reward + (self.gamma * self.get_maxQ(state)))
        return