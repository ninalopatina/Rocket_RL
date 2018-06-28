#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:21:30 2018

@author: ninalopatina
"""

import os
import pandas as pd
pd.set_option("display.max_columns",30)

plotting = True


WD = os.listdir()

#!/usr/bin/env python
from stat import S_ISREG, ST_CTIME, ST_MODE
import os, sys, time

import pickle
import os
import yaml

import matplotlib.pyplot as plt

# ##TO DO: get cfg in here more elegantly
#set config path
config_dir = 'config/'
CWD_PATH = os.getcwd()
config_path = os.path.join(CWD_PATH,config_dir,"config.yml")

with open(config_path, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# path to the directory (relative or absolute)
dirpath = '/Users/ninalopatina/ray_results/RocketRL/'#sys.argv[1] if len(sys.argv) == 2 else r'.'

# get all entries in the directory w/ stats
entries = (os.path.join(dirpath, fn) for fn in os.listdir(dirpath))
entries = ((os.stat(path), path) for path in entries)

# leave only regular files, insert creation date
entries = ((stat[ST_CTIME], path)
           for stat, path in entries)# if S_ISREG(stat[ST_MODE]))
#NOTE: on Windows `ST_CTIME` is a creation date 
#  but on Unix it could be something else
#NOTE: use `ST_MTIME` to sort by a modification date
#
for cdate, p in sorted(entries):
    if 'DS_Store' not in p:
        path = p
        print(p)
#    
#   
    
def save_plot(fig,title):
    save_dir = os.path.join(CWD_PATH, cfg['result_path'],cfg['model_result_path'])
    fig.savefig(save_dir + title+".png")
#now path is the last one
#df = pd.read_csv(path+'/progress.csv')
df = pd.read_json(path+'/result.json',lines=True)

df['timesteps_per_second'] = df['timesteps_this_iter']/df['time_this_iter_s']

#use the below to plot all numerical values
exclude = ['config', 'date','experiment_id','hostname','info','timestamp','node_ip','done']
include = ['episode_len_mean','episode_reward_mean','timesteps_per_second']

df_info = pd.DataFrame()
for row in df.index:
    df2 = pd.DataFrame.from_dict(data = df['info'][row],orient='index')
    df2 = df2.transpose()
    df_info = df_info.append(df2)
    
df_info['training_iteration'] = df['training_iteration'].values

if plotting == True:
    for var in df.columns:
        if var in include:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            df.plot.line(x='training_iteration', y = var,legend = False, ax = ax)
            
            if 'reward' in var:
                var = 'Episode Reward Mean'
            elif 'len' in var:
                var = 'Mean Steps to Target'
            ax.set_title(var+' Over During Training')
            ax.set_ylabel(var)
            ax.set_xlabel('Training Iteration')
            save_plot(fig,var)
            
            
            
        

#
#    for var in df_info.columns:
#        df_info.plot.scatter(x='training_iteration', y = var)
#        

##plot info values

#    for k,v in df['info']