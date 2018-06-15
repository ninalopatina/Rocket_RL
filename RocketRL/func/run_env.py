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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 
import yaml

#import an agent for this
#import model.agent.random_agent as random_agent

#set home directory
home_dir = '/Users/ninalopatina/Desktop/Rocket_RL/'
code_dir = 'src/python/'
config_dir = 'config/'

os.chdir(home_dir + config_dir)

#import cfg variables
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


#load my custom environment
env = gym.make('SimpleCorridor-v0')

#TO DO: test that the environment has right dimensions:

state = env.reset() #start at the beginning

new_states = []
rewards = []
dones = []
pos = 0

Q = np.zeros([env.observation_space.n[0], env.action_space.n])
G = 0
gamma = 0.618

df_experiment = pd.DataFrame(columns = ['Total Reward','Num steps'])

for episode in range(1,101):
    done = False
    G, reward, counter = 0,0,0
    state = env.reset()

    #make a dataframe to look at an episode
    df = pd.DataFrame(columns = ['Input Temp','Output Temp','Action','Reward'])#,'Total Reward'])
    
    while done != True:
        
            #from your current state, get input temp:
            in_state = int([state][0][0][0])

            #use amax instead of argmax in case 2 actions have equal value
            winner = np.argwhere(Q[in_state] == np.amax(Q[in_state]))
            action = random.choice(winner) #in case there are tied q-values
            
            #TO DO: move to cfg
            action_map = {0:1,1:2,2:3,3:4,
                      4:-1,5:-2,6:-3,7:-4}
            act = action[0]
            
            #take your action:
            state2, reward, done = env.step(action)
            
            
            #track the start state and action for this step
            #multiply by 10 to reflect actual temp
            df.at[counter,'Input Temp'] = float(state[0][0]*10)
            df.at[counter,'Output Temp'] = float(state[0][1]*10)
            df.at[counter,'Action'] = float(action_map[act]*10) 
            df.at[counter,'Reward'] = float(reward) #track reward you got for your action 
            
            #get input state after your action:
            in_state2 = int([state2][0][0][0])
            
            #attribute the Q-value for that state,action from the reward you got
            Q[in_state,action] = (reward + gamma * np.max(Q[in_state2]))
            G += reward
            counter += 1
            state = state2 
    if episode == 1:
        df1 = df.copy() #save the first run
    
    df_experiment.at[episode,'Total Reward'] = float(G)
    df_experiment.at[episode,'Num steps'] = counter
    
    if episode % 50 == 0:
        print('Episode {} Total Reward: {} counter: {}'.format(episode,G,counter))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 18:08:26 2018

@author: ninalopatina
"""
colors = {'Total Reward': 'k','Reward':'b','episodes':'g','Input Temp':'m','Output Temp':'r','Num steps':'k','Action':'g'}

def plot_results(cfg,DF,sup_title,y_label, fig, axes):
    n_var = DF.shape[1] 
    if sup_title == 'Results over episodes':
        fig, axes = plt.subplots(n_var,1,figsize=[5,n_var*1.55])
    else:
        fig, axes = plt.subplots(n_var-1,1,figsize=[5,n_var*1.55])

    
    plt.subplots_adjust(hspace = .05, right = 0.8, bottom = 0.2)

    fig.suptitle(sup_title,fontsize =20 )
    
    for i ,varr in enumerate(DF.columns):
        plot_var(DF.copy(), i,varr, sup_title,y_label,axes)
    
    save_dir = (cfg['home_dir'] + cfg['fig_dir'])    
    fig.savefig(save_dir + sup_title+".png")
    return fig, axes
    

def plot_var(DF_plot, i,varr, sup_title,x_label,axes):
    DF_plot['index'] = DF_plot.index
    color = colors[varr]
    if sup_title == 'First episode':
        linestyle = 'dotted'

    else:
        linestyle = 'solid'
    if sup_title != 'Results over episodes':
        if (i == 0) | (i == 1) :
            ax_num = 0
        else:
            ax_num = i-1
    else:
        ax_num = i
    ax = axes[ax_num]
    
    if  'Temp' in varr:
        linee = DF_plot.plot(x='index', y = varr, ax = ax, color = color, linestyle = linestyle)
           
    else:
        linee = DF_plot.plot(x='index', y = varr, ax = ax, color = color, linestyle = linestyle,legend=None)

    ax.set_xlabel(x_label, fontsize =12)
    
    ax.set_ylabel(varr, fontsize =12)
    
    if 'Action' in varr:
        ax.set_ylim(-50,50)
        ax.set_yticks([-40,-20,0,20,40])
        ax.set_xlim(-10,550)
        xmin = -10
        xmax = 550
   
    if 'Temp' in varr:
        ax.set_ylabel('Temp', fontsize =12)
        ax.set_ylim(0,800)
        xmin = -10
        xmax = 550

        ax.set_xlim(xmin,xmax)
        start = 150
        goal = 650
        offset = -310
        ax.hlines(y=start, xmin=xmin, xmax=xmax, colors='y', linestyles= 'dashed')
        ax.annotate('Start',[xmax+offset,start-20],size = 12)
        ax.hlines(y=goal, xmin=xmin, xmax=xmax, colors='c', linestyles= 'dashed')
        ax.annotate('Goal',[xmax+offset,goal-20],size = 12)        
               
    if varr =='Reward':
        ax.set_ylim(-5500,250)
        xmin = -10
        xmax = 550
        ax.set_xlim(xmin,xmax)
        ax.set_ylabel(varr, fontsize =12)
        
    ax.yaxis.tick_right()

    
fig = 0
axes = 0

fig, axes = plot_results(cfg,df_experiment,'Results over episodes','episode',fig=fig, axes =axes)

fig, axes = plot_results(cfg,df1,'First episode','step',fig=fig, axes =axes)

fig, axes = plot_results(cfg,df,'Last episode','step',fig=fig, axes =axes)
  


