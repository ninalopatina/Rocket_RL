"""This code is adapted from rllib's example of a custom env"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym import spaces
from gym.envs.registration import EnvSpec

# ##TO DO: get cfg in here more elegantly
#set config path
# config_dir = 'config/'
# CWD_PATH = os.getcwd()
# config_path = os.path.join(CWD_PATH,config_dir,"model.yml")
#
# with open(config_path, 'r') as ymlfile:
#     cfg = yaml.load(ymlfile)

# import ray
# from ray.tune import run_experiments
# from ray.tune.registry import register_env

#TO DO: figure out how to add this to github repo

class SimpleTemp(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self,temp_knob = 1,min_in_temp = 0, max_in_temp = 80,
     MSE_thresh = 2,action_space=8):#, config):
        def temp_func(x):
            y = self.m * x + self.c
            return y
        #TO DO remove hard coding here to make this dynamic

        #all these values are divided by 10 to make this simpler:

        #temp change:
        self.temp_knob = temp_knob

        #relationship between input & output temp:
        #output temp = input temp * m + c
        self.c = 14
        self.m = 0.66

        #temperature range
        self.min_in_temp = min_in_temp #min temp
        self.max_in_temp = max_in_temp #max temp

        self.min_out_temp = int(temp_func(self.min_in_temp)) #min temp
        self.max_out_temp = int(temp_func(self.max_in_temp)) #max temp

        #starts and goals:

        self.MSE_thresh = MSE_thresh
        # self.MSE_thresh2 = 5

        #2d state space
        self.low_state = np.array([self.min_in_temp, self.min_out_temp, self.min_out_temp])
        self.high_state = np.array([self.max_in_temp, self.max_out_temp, self.max_out_temp])

#        #2d state space
#        self.low_state = np.array([self.min_out_temp, self.min_out_temp])
#        self.high_state = np.array([self.max_out_temp, self.max_out_temp])

        #how much each change in input temp changes output temp

#        self.end_pos = self.goal_temp #config["corridor_length"]
#        self.cur_pos = self.start_temp
        self.action_space = spaces.Discrete(action_space)

        self.observation_space = spaces.Tuple((spaces.Discrete(self.max_in_temp-self.min_in_temp), spaces.Discrete(self.max_out_temp-self.min_out_temp), spaces.Discrete(self.max_out_temp-self.min_out_temp)))

#        Discrete((self.high_state-self.low_state).astype(int),)#, shape=(2,), dtype=np.float32)
        #TO DO: Uncomment below to make spaces continuous
        #        self.action_space = Box(low=self.min_action, high=self.max_action, shape=(1,))
#        self.observation_space = Box(
#            low=self.low_state, high=self.high_state)#, shape=(2,), dtype=np.float32)

        #TO DO: below needs to be fixed.. idk what the format is of
        self._spec = EnvSpec("SimpleCorridor-{}-v0".format(1))

    def reset(self): #start over
        def temp_func(x):
            y = self.m * x + self.c
            return y

        #on every reset, you have a new goal temp:
        self.goal_temp = np.random.randint(self.min_out_temp+1,self.max_out_temp)
        self.start_temp = np.random.randint(self.min_in_temp,self.max_in_temp)

#        print('start',self.start_temp)
#        print('goal',self.goal_temp)

        self.state = np.array([self.start_temp,self.goal_temp,temp_func(self.start_temp)])
        return [self.state]

    def step(self, action):
        #TO DO: Move this
        def temp_func(x):
            y = self.m * x + self.c
            return y

        in_temp = self.state[0]
#        out_temp = self.state[2]

        act = action[0]
        #increase or decrease the input temp





        new_temp = in_temp + self.temp_knob*cfg['action_map'][act]

                #is this temp change viable?
        if (new_temp <= self.min_in_temp):
            new_temp = self.min_in_temp+1
            in_temp = new_temp

        elif (new_temp >= self.max_in_temp):
            new_temp = self.max_in_temp-1
            in_temp = new_temp

        else:
            in_temp += self.temp_knob*cfg['action_map'][act]


        #get the corresponding output temp:
        out_temp = temp_func(in_temp)

        #get the MSE for reward function
        MSE = (self.goal_temp-out_temp)**2

        #update your state
        self.state = np.array([in_temp,self.goal_temp, out_temp])

        done = MSE <= self.MSE_thresh

        #get the corresponding reward
        reward = 0
        if done:
            reward += 100.0
#        elif MSE <= self.MSE_thresh2:
#            reward =+ 50
        elif MSE > self.MSE_thresh:
            reward -= MSE



        return [self.state], reward, done
