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

#To do: set this up so these are set in terminal, and the defaults are contained in a function here
#set main params;
plot_data = 0 #0 is none, 1 is all, 2 is 2d only, 3 is 3d
run_RL = True
run_regression = False
save_regression = False

#import the functions from the functions file
import func.data_processing as RocketData
import func.run_env as RocketRunEnv

#TO do: set up class and run like this:
#rocketData = RocketData()
#RocketData.plotter('2d')

#TO DO: pick which to plot and turn the rest off by default
if (plot_data>0) | (run_regression == True):
    df_data, approx = RocketData.data_process(plot_data,run_regression,save_regression)

if run_RL == True:
    #Run your agent in the custom environment:
    Q,df, df1, df_experiment,df_time = RocketRunEnv.test_agent()
    
