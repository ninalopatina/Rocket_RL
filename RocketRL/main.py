#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:49:31 2018

This code is for an RL agent that determines the optimal input parameters for 
the desired ouput for a 3d-printed rocket engine component. 

Stage 0: load and visualize data, research past approaches to similar problems

Stage 1: Implement an offline agent to learn optimal parameters from simulation data, 
using shallow RL

Stage 2: Above with keras-rl DQN and deep SARSA

Stage 3: Bootstrap above with an engineer's inputs for parameters to test

Stage 4: RL agent learns the optimal input parameters

Stage 5: Parallelize and asynchonize the above


Final Stage: what should ISP (measure of thrust) look like for a particular problem


@author: ninalopatina
"""
import os 

#set home directory
home_dir = '/Users/ninalopatina/Desktop/Rocket_RL/'
code_dir = 'src/python/'
config_dir = 'config/'

import yaml

import pandas as pd
pd.set_option("display.max_columns",50)

os.chdir(home_dir + config_dir)


with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

#TO DO: is this kosher? 
all_var = cfg['in_var'].copy()
for item in cfg['out_var']:
    all_var.append(item)
cfg['all_var']= all_var

#set running params
RL = 0


#import the functions from the functions file
os.chdir(home_dir + code_dir)
import data.util as RocketData

#rocketData = RocketData() 

df, df_mini = RocketData.data_process(cfg,plotting2d = True)
#RocketData.plotter('2d') ##QUESTION: does cfg go here? 

