#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:49:31 2018

This codebase creates an RL agent that determines input parameters that reach
the desired ouputs for tuning a rocket engine component.

The main functions are:
    1) Linear regression to map inputs --> outputs based on some cached flow
    simulation data from Fluent.
    2) Reinforcement Learning to derive inputs from target outputs.
    3) Plotting some metrics of the training progress. This can be run
    concurrently while training the agent.


@author: ninalopatina
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import the Rocket_RL functions
import Rocket_RL.python.func.data_processing as RocketData
import Rocket_RL.python.func.ray_funcs as RF

# Import a few other funcs for the main script
import argparse, os, yaml

#imports for RLLib
import ray
from ray.tune import run_experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-rreg','--run_reg', dest = 'run_regression', type = bool,
                        default = True, help = 'True to run the regression')
    # IMPORTANT NOTE: If you turn on the below and save the regression, make sure
    # you do not overwrite it before rolling out your policy. This should only be
    # toggled on or overwritten when training the model. The identical regression
    # coefficients have to be used when rolling out as training.
    parser.add_argument('-sreg','--save_reg', dest = 'save_regression', type = bool,
                        default = False, help = 'True to save the regression')
    parser.add_argument('-rRL','--run_RL', dest = 'run_RL', type = bool,
                        default = True, help = 'True to run RL model')
    parser.add_argument('-pRL','--plot_RL', dest = 'plot_RL', type = bool,
                        default = True, help = 'True to plot RL results')
    parser.add_argument('-plot','--plot_data', dest = 'plot_data', type = int,
                        default = False, help = 'True to plot Fluent simulation data in 3d')
    parser.add_argument('-cfg','--config', dest = 'config_dir', type = str,
                        default = 'Rocket_RL/config/config.yml', help = 'where the config file is located')
    parser.add_argument('-rayinit','--rayinit', dest = 'rayinit', type = bool,
                        default = True, help = 'should only be init the first time you run if running in an IDE')
    args = parser.parse_args()

    # Set config path & load the config variables.
    CWD_PATH = os.getcwd()
    config_path = os.path.join(CWD_PATH,args.config_dir)
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Add the current working path to the config var.
    cfg['CWD_PATH'] = CWD_PATH

    # If importing new data, set plot_data to true to take a glance at it to verify
    # if it looks reasonable.
    if (args.plot_data==True) | (args.run_regression == True):

        RocketData.data_process(cfg,args.plot_data,args.run_regression,args.save_regression)

    if args.run_RL == True:
        if args.rayinit == True:
            ray.init()
        run_experiments({
            "RocketRL": {
                "run": cfg['agent'],# Which agent to run
                #conditions under which you would stop:
                "stop": {"time_total_s": cfg['time_total_s'], "timesteps_total": cfg['timesteps_total'], "episode_reward_mean": cfg['episode_reward_mean']},
                #The custom environment:
                "env": "AllVar-v0",
                #Checkpoint on every iteration:
                "checkpoint-freq": cfg['checkpoint-freq'],
                #Num workers has to be 1 fewer than the available CPU!!!!!
                "config": {
                    "num_workers": cfg['num_workers'],
                    "gamma": cfg['gamma'],
                    "horizon": cfg['horizon'],
                    "num_sgd_iter": cfg['num_sgd_iter'],
                    "sgd_stepsize": cfg['sgd_stepsize'],
                    "timesteps_per_batch": cfg['timesteps_per_batch'],
                    "min_steps_per_task": cfg['min_steps_per_task']
                },
            },
        })

    if args.plot_RL == True:
        df_model = RF.rllib_plot(cfg)
