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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import the functions from the functions file
import func.data_processing as RocketData

from gym.envs.RocketRL.RocketEnv_2T import TwoTemp

#import a few other funcs for the main script
import argparse
import os
import yaml

import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import pickle

import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env

#to run: python main.py -run True -plot False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-reg','--run_reg', dest = 'run_regression', type = int, 
                        default = 2, help = '1 to run the regression, 2 to save it. 0 neither.')   
    parser.add_argument('-RL','--run_RL', dest = 'run_RL', type = bool, 
                        default = True, help = 'True to run RL')   
    parser.add_argument('-plot','--plot_data', dest = 'plot_data', type = int, 
                        default = 0, help = '0 is none, 1 is all, 2 is 2d only, 3 is 3d')
    parser.add_argument('-cfg','--config', dest = 'config_dir', type = str, 
                        default = 'config/config.yml', help = '1 to run the regression, 2 to save it. 0 neither.')
    parser.add_argument('-rayinit','--rayinit', dest = 'rayinit', type = bool, 
                        default = True, help = 'should only be init the first time you run')                        
    args = parser.parse_args()
    
    #set config path
    CWD_PATH = os.getcwd()
    config_path = os.path.join(CWD_PATH,args.config_dir)
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    #add the current working path to your config var
    cfg['CWD_PATH'] = CWD_PATH
    #add a few other cfg var
    cfg = RocketData.cfg_mod(cfg)

    #TO do: set up class and run like this:
    #rocketData = RocketData()
    #RocketData.plotter('2d')
    
        
    RocketData.labelmaker(cfg,'labels1')
    RocketData.labelmaker(cfg,'labels2')
    
    
    #TO DO: pick which to plot and turn the rest off by default
    if (args.plot_data>0) | (args.run_regression >0):
        df_data, approx, model = RocketData.data_process(cfg,args.plot_data,args.run_regression)

    if args.run_RL == True:
        env_creator_name = "TwoTemp"
        register_env(env_creator_name, lambda config: TwoTemp(config))
        #TO DO: make this switch on/off if youve already init'd, w/   ray.disconnect()  ?
        if args.rayinit == True:
            ray.init()
        run_experiments({
            "RocketRL": {
                "run": "PPO",#Which agent
                #conditions under which you would stop:
                "stop": {"time_total_s": 6000, "timesteps_total": 1000000, "episode_reward_mean": 100000},
                "env": "TwoTemp",
                "checkpoint-freq": 1,
                "config": {
                    "num_workers": 3,
         
        
                },
            },
        })
