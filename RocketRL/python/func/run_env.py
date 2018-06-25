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
import datetime
import pickle

import func.RL_results as RocketPlot

#set config path
config_dir = 'config/'
CWD_PATH = os.getcwd()
config_path = os.path.join(CWD_PATH,config_dir,"model.yml")

with open(config_path, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

#import an agent for this
#import model.agent.random_agent as random_agent

def test_agent():
    #load my custom environment
    env_type = 'TwoTemp-v0' #SimpleTemp or TwoTemp-v0
    env = gym.make(env_type)

    #TO DO: test that the environment has right dimensions:
    #start at the beginning
    #    state = env.reset()

    #TRAIN:::
    #initialize Q-values & Total reward (G):

    if cfg['training'] == True:
        if env_type == 'TwoTemp-v0':
            Q = np.zeros([env.observation_space.spaces[0].n,env.observation_space.spaces[1].n,env.observation_space.spaces[2].n, env.action_space.n])
        else:
            Q = np.zeros([env.observation_space.spaces[0].n,env.observation_space.spaces[1].n, env.action_space.n])
            
        Q,df, df1, df_experiment,df_time = run_temp(Q,env,env_type)
        
        #pickle these outputs in case you want to run this model later
        pickle_name = (str(env_type)+str(cfg['n_exp'])+str(datetime.datetime.now())+'.p')
        pickle_path = os.path.join(CWD_PATH,cfg['result_path'],cfg['pickle_path'],pickle_name)
        pickle.dump([Q,df, df1, df_experiment,df_time], open(pickle_path, "wb" ) )

        RocketPlot.plot_results(df, df1, df_experiment)

        time_test('ad',df_time)
        time_test('bc',df_time)
        
    else:
        ###TEST:::
        #to test, don't intialize Q:
        Q = pickle.load( open(pickle_path4, "rb" ) )
        print(np.sum(Q))

        Q,df, df1, df_experimentT,df_time = run_temp(Q,env,env_type)
        df_experiment = pd.read_pickle(pickle_path3)
        RocketPlot.plot_results(df, df1, df_experiment)
#
    time_test('ad',df_time)
    time_test('bc',df_time)

    return Q,df, df1, df_experiment,df_time


def time_test(val,dft):
    delts = dft[val].values
    mn = np.mean(delts[1:])/1000

    print('average loop microseconds',val,':', mn)


def run_temp(Q,env,env_type):
    G = 0
    #Make a dataframe to track progress over the course of experiments:

    TR = []
    NS = []

    ad = []
    bc = []

    if cfg['training'] == True:
        n_exp = cfg['n_exp']
    else:
        n_exp = 2

    for episode in range(1,n_exp):
        #make a dataframe to store data from each episode
#        df = pd.DataFrame(columns = ['Input Temp','Output Temp','Action','Reward'])#,'Total Reward'])

        done = False
        G, reward, counter = 0,0,0
        state = env.reset()

        IT1 = []
        IT2 = []
        OT= []
        T = []
        A = []
        R = []

        while done != True:
                a = datetime.datetime.now()

                #from your current state, get input temp:
                if env_type == 'TwoTemp-v0':
                    #for the purpose of our q-values, we're re-setting the q-values where index 0 is
                    #the min temp
                    in_state_1 = int(state[0][0])- env.low_state[0]
                    in_state_2 = int(state[0][1])- env.low_state[1]
                    target = int(state[0][2]) - env.low_state[2]
                    #use amax instead of argmax in case 2 actions have equal value
                    winner = np.argwhere(Q[in_state_1,in_state_2,target] == np.amax(Q[in_state_1,in_state_2,target]))
                    action = random.choice(winner) #in case there are tied q-values
                        
                else:
                    #for the purpose of our q-values, we're re-setting the q-values where index 0 is
                    #the min temp
                    in_state = int([state][0][0][0])- env.low_state[0]
                    target = int(state[0][1]) - env.low_state[1]
                    #use amax instead of argmax in case 2 actions have equal value
                    winner = np.argwhere(Q[in_state,target] == np.amax(Q[in_state,target]))
                    action = random.choice(winner) #in case there are tied q-values
      
                act = action[0] #data array workaround

                #take your action:
                b = datetime.datetime.now()
                state2, reward, done = env.step(action)
                c = datetime.datetime.now()

                #track the start state and action for this step
                #multiply by 10 to reflect actual temp
                if (episode == 1) | (episode == n_exp-1): #save results
                    IT1.append(float((state[0][0])*10))
                    if env_type == 'TwoTemp-v0':
                        IT2.append(float((state[0][1])*10))
                        OT.append(float((state[0][3]*10)))
                        T.append((target + env.low_state[2])*10)
                        A.append(act)
                                
                    else:
                        OT.append(float(state[0][1]*10))
                        T.append((target + env.low_state[1])*10)
                        A.append(float(cfg['action_map'][act]*10))
                        in_state2 = int([state2][0][0][0])
               
                    R.append(float(reward)) #track reward you got for your action

                #get input state after yours action:
                if env_type == 'TwoTemp-v0':
                    in_state2_1 = int(state2[0][0])- env.low_state[0]
                    in_state2_2 = int(state2[0][1])- env.low_state[1]


                #attribute the Q-value for that state,action from the reward you got
#                if training ==True:
                if env_type == 'TwoTemp-v0':
                    Q[in_state_1, in_state_2,target,action] = (reward + cfg['gamma'] * np.max(Q[in_state2_1,in_state2_2,target]))
                else:
                    Q[in_state,target,action] = (reward + cfg['gamma'] * np.max(Q[in_state2,target]))
                G += reward
                counter += 1
                state = state2

                if counter>cfg['max_steps']:
                    break

                d = datetime.datetime.now()

                ad.append(d-a)
                bc.append(c-b)

        df = pd.DataFrame(columns = ['Input Temp1','Input Temp2','Output Temp','Target','Action','Reward'],
                          data = np.array([IT1,IT2,OT,T,A,R]).transpose())
        if counter>cfg['max_steps']:#
            print('episode maxed out', episode)

        if episode == 1:
            df1 = df.copy() #save the first run

        TR.append(float(G))
        NS.append(counter)

        if episode % 10 == 0:
            print('Episode {} Total Reward: {} counter: {}'.format(episode,G,counter))

    df_experiment = pd.DataFrame(columns = ['Total Reward','Num steps'],data = np.array([TR,NS]).transpose())
    df_time = pd.DataFrame(columns = ['ad','bc'],data = np.array([ad,bc]).transpose())

    return Q, df, df1, df_experiment,df_time
