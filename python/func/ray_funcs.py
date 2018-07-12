#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:21:30 2018

@author: ninalopatina
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from stat import ST_CTIME
# SEt some visualization options
pd.set_option("display.max_columns",30)
plt.style.use('ggplot')

def rllib_plot(cfg):
    """"
    This function opens the data RLlib saved from the last experiment and plots a few variables of interest.
    """
    df = open_file(cfg)
    df = mod_df(df)
    plot_var(cfg,df)
    return df

def save_plot(cfg,fig,title):
    """
    This function saves the plot.
    """
    save_dir = os.path.join(cfg['CWD_PATH'],cfg['repo_path'], cfg['result_path'],cfg['model_result_path'])
    fig.savefig(save_dir + title+".png")

def open_file(cfg):
    """
    Open the results file from the most recent experiment.
    """
        
    dirpath = os.path.join(cfg['CWD_PATH'], cfg['ray_results_path'])
    
    # get all entries in the directory w/ stats
    entries = (os.path.join(dirpath, fn) for fn in os.listdir(dirpath))
    entries = ((os.stat(path), path) for path in entries)
    
    # leave only regular files, insert creation date
    entries = ((stat[ST_CTIME], path)
               for stat, path in entries)
    #NOTE: on Windows `ST_CTIME` is a creation date 
    #  but on Unix it could be something else
    #NOTE: use `ST_MTIME` to sort by a modification date

    #to do: get just the last one
    for cdate, p in sorted(entries)[:]:
        if 'DS_Store' not in p:
            path = p
        
    df = pd.read_json(path+'/result.json',lines=True)
    return df

def mod_df(df):
    """
    Add features to plot
    """
    df['timesteps_per_second'] = df['timesteps_this_iter']/df['time_this_iter_s']
    return df


def plot_var(cfg,df):
    """
    Plot variables as determined below
    """
    exclude = ['config', 'date','experiment_id','hostname','info','timestamp','node_ip','done']
    include = ['episode_len_mean','episode_reward_mean','timesteps_per_second']
    
    df_info = pd.DataFrame()
    for row in df.index:
        df2 = pd.DataFrame.from_dict(data = df['info'][row],orient='index')
        df2 = df2.transpose()
        df_info = df_info.append(df2)
        
    df_info['training_iteration'] = df['training_iteration'].values
  
    f_scale = .85
    fs_label = 20*f_scale
    fs_title = 24*f_scale
    fs_ticks = 14*f_scale
    
    for var in df.columns:
        if var in include:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            df.plot.line(x='training_iteration', y = var,legend = False, ax = ax)#,color='r')
            
            if 'reward' in var:
                var = 'Episode Reward Mean'
            elif 'len' in var:
                var = 'Mean Steps to Target'
#                ax.set_ylim(0,1100)
            ax.set_title(var+' by Iteration',fontsize = fs_title)
            ax.set_ylabel(var,fontsize = fs_label)
            ax.set_xlabel('Training Iteration',fontsize = fs_label)
            ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
            save_plot(cfg,fig,var)
    return df