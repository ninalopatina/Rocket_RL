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
code_dir = 'code/'

#import the functions from the functions file
os.chdir(home_dir + code_dir)
import RocketRL_funcs_data as RocketData

RocketData.data_process()