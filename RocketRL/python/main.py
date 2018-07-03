#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:49:31 2018

This code is for an RL agent that determines the optimal input parameters for
the desired ouput to tune a rocket engine component.

@author: ninalopatina
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import the functions from the functions file
import func.data_processing as RocketData

#to do: fix drive path issue to import this:
#import func.ray_funcs as RF 

#import a few other funcs for the main script
import argparse
import os
import yaml

#imports for RLLib
import ray
from ray.tune import run_experiments

##to delete after I sort out the gym registation:
#from gym.envs.RocketRL.RocketEnv import AllVar
#from gym.envs.registration import EnvSpec
#from ray.tune.registry import register_env

#to run: python main.py -run True -plot False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-rreg','--run_reg', dest = 'run_regression', type = bool,
                        default = True, help = 'True to run the regression')
    # IMPORTANT NOTE: If you turn on the below and save the regression, make sure
    # you do not overwrite it before rolling out your policy. This should only be 
    # toggled on or overwritten when training the model. The identical regression
    # coefficients have to be used when rolling out or it won't work. 
    parser.add_argument('-sreg','--save_reg', dest = 'save_regression', type = bool,
                        default = False, help = 'True to save the regression')
    parser.add_argument('-rRL','--run_RL', dest = 'run_RL', type = bool,
                        default = True, help = 'True to run RL model')
    parser.add_argument('-pRL','--plot_RL', dest = 'plot_RL', type = bool,
                        default = True, help = 'True to plot RL results')
    parser.add_argument('-plot','--plot_data', dest = 'plot_data', type = int,
                        default = False, help = 'True to plot the data in 3d')
    parser.add_argument('-cfg','--config', dest = 'config_dir', type = str,
                        default = 'config/config.yml', help = 'where the config file is located')
    parser.add_argument('-rayinit','--rayinit', dest = 'rayinit', type = bool,
                        default = True, help = 'should only be init the first time you run if running in an IDE')
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

    if (args.plot_data==True) | (args.run_regression == True):
        # If importing new data, run the below to take a glance at it to verify
        # if it looks reasonable.
        RocketData.data_process(cfg,args.plot_data,args.run_regression,args.save_regression)

    if args.run_RL == True:
        ##to delete after I sort out the gym registation:
#        env_creator_name = "AllVar"
#        register_env(env_creator_name, lambda config: AllVar(config))      
        if args.rayinit == True:
            ray.init()
        run_experiments({
            "RocketRL": {
                "run": "PPO",# Which agent to run
                #conditions under which you would stop:
                "stop": {"time_total_s": 600000, "timesteps_total": 100000000, "episode_reward_mean": 100000},
                #The custom environment: 
                "env": "AllVar-v0",
                #Checkpoint on every iteration:
                "checkpoint-freq": 1,
                #Num workers has to be 1 fewer than the available CPU!!!!!
                "config": {
                    "num_workers": 3,
                },
            },
        })