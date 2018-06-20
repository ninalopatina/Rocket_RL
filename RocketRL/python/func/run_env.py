#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:20:23 2018

@author: ninalopatina
"""

import gym
import pandas as pd
import random
import numpy as np
import os
import yaml

#set config path
config_dir = 'config/'
CWD_PATH = os.getcwd()
config_path = os.path.join(CWD_PATH,config_dir,"model.yml")

with open(config_path, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

#import an agent for this
#import model.agent.random_agent as random_agent


def run_temp():
    #load my custom environment
    env = gym.make('SimpleTemp-v0')

    #TO DO: test that the environment has right dimensions:
    #start at the beginning
#    state = env.reset()

    #initialize Q-values & Total reward (G):
    Q = np.zeros([env.observation_space.spaces[0].n,env.observation_space.spaces[1].n, env.action_space.n])
    G = 0

    #Make a dataframe to track progress over the course of experiments:
    df_experiment = pd.DataFrame(columns = ['Total Reward','Num steps'])

    for episode in range(1,cfg['n_exp']):
        #make a dataframe to store data from each episode
        df = pd.DataFrame(columns = ['Input Temp','Output Temp','Action','Reward'])#,'Total Reward'])

        done = False
        G, reward, counter = 0,0,0
        state = env.reset()

        while done != True:

                #from your current state, get input temp:
                in_state = int([state][0][0][0])
                #for the purpose of our q-values, we're re-setting the q-values where index 0 is
                #the min temp
                target = int(state[0][1]) - env.low_state[1]

                #use amax instead of argmax in case 2 actions have equal value
                winner = np.argwhere(Q[in_state,target] == np.amax(Q[in_state,target]))
                action = random.choice(winner) #in case there are tied q-values

                act = action[0] #data array workaround

                #take your action:
                state2, reward, done = env.step(action)

                #track the start state and action for this step
                #multiply by 10 to reflect actual temp
                df.at[counter,'Input Temp'] = float(state[0][0]*10)
                df.at[counter,'Output Temp'] = float(state[0][2]*10)

                df.at[counter,'Target'] = (target + env.low_state[1])*10
                df.at[counter,'Action'] = float(cfg['action_map'][act]*10)
                df.at[counter,'Reward'] = float(reward) #track reward you got for your action

                #get input state after your action:
                in_state2 = int([state2][0][0][0])

                #attribute the Q-value for that state,action from the reward you got
                if in_state2>=80:
                    print(in_state2,in_state,target,action)
                Q[in_state,target,action] = (reward + cfg['gamma'] * np.max(Q[in_state2,target]))
                G += reward
                counter += 1
                state = state2

                if counter>1000:
#                    print(in_state2,in_state,target,action)
                    df.plot()
                    break

        if episode == 1:
            df1 = df.copy() #save the first run

        df_experiment.at[episode,'Total Reward'] = float(G)
        df_experiment.at[episode,'Num steps'] = counter

        if episode % 50 == 0:
            print('Episode {} Total Reward: {} counter: {}'.format(episode,G,counter))
    return df, df1, df_experiment
