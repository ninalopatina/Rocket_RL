"""This code is adapted from rllib's example of a custom env"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym.spaces import Discrete, Box
from gym.envs.registration import EnvSpec

##TO DO: get cfg in here more elegantly 
#import your cfg file:
import yaml

#set home directory
home_dir = '/Users/ninalopatina/gitrepos/Rocket_RL/'
config_dir = 'config/'

with open(home_dir + config_dir+"config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# import ray
# from ray.tune import run_experiments
# from ray.tune.registry import register_env

#TO DO: figure out how to add this to github repo



class SimpleTemp(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self):#, config):
        #TO DO remove hard coding here to make this dynamic

        #all these values are divided by 10 to make this simpler:

        #temp change:
        self.temp_knob = 1

        #temperature range
        self.min_position = 0 #min temp
        self.max_position = 80 #max temp

        #starts and goals:
        self.goal_temp = 67#np.random.randint(self.min_position,self.max_position)
        self.start_temp = 2#np.random.randint(self.min_position,self.max_position)
        self.MSE_thresh = 2
        self.MSE_thresh2 = 5

        #2d state space
        self.low_state = np.array([self.min_position, self.min_position])
        self.high_state = np.array([self.max_position, self.max_position])

        print('start',self.start_temp)
        print('goal',self.goal_temp)

        #relationship between input & output temp:
        #output temp = input temp * m + c
        self.c = 14
        self.m = 0.66

        #how much each change in input temp changes output temp

        self.end_pos = self.goal_temp #config["corridor_length"]
        self.cur_pos = self.start_temp
        self.action_space = Discrete(8)

        self.observation_space = Discrete(
             self.high_state-self.low_state,)#, shape=(2,), dtype=np.float32)
        #TO DO: Uncomment below to make spaces continuous
        #        self.action_space = Box(low=self.min_action, high=self.max_action, shape=(1,))
#        self.observation_space = Box(
#            low=self.low_state, high=self.high_state)#, shape=(2,), dtype=np.float32)

        self._spec = EnvSpec("SimpleCorridor-{}-v0".format(self.end_pos))

    def reset(self): #start over
        def temp_func(x):
            y = self.m * x + self.c
            return y

        self.state = np.array([self.start_temp,temp_func(self.start_temp)])
        return [self.state]

    def step(self, action):
        #TO DO: Move this
        def temp_func(x):
            y = self.m * x + self.c
            return y

        in_temp = self.state[0]
        out_temp = self.state[1]

        act = action[0]
        #increase or decrease the input temp

        #is this temp change viable?
        new_temp = in_temp + self.temp_knob*cfg['action_map'][act]
        if (new_temp > self.min_position) & (new_temp < self.max_position):

            in_temp += self.temp_knob*cfg['action_map'][act]

            #get the corresponding output temp:
            out_temp = temp_func(in_temp)

            #get the MSE for reward function
            MSE = (self.goal_temp-out_temp)**2

            #update your state
            self.state = np.array([in_temp, out_temp])

            done = MSE <= self.MSE_thresh

            #get the corresponding reward
            reward = 0
            if done:
                reward += 100.0
    #        elif MSE <= self.MSE_thresh2:
    #            reward =+ 50
            elif MSE > self.MSE_thresh:
                reward -= MSE
    #        reward-= math.pow(action[0],2)*0.1
        else: #cannot make this action
            reward = 0
            reward -= 5000 # don't do this action
            done = False

        return [self.state], reward, done
