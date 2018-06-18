#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 18:08:26 2018

@author: ninalopatina
"""
import gym
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

def plot_each(cfg,DF,sup_title,y_label, fig, axes):
    n_var = DF.shape[1] 
    if sup_title == 'Results over episodes':
        fig, axes = plt.subplots(n_var,1,figsize=[5,n_var*1.55])
    else:
        fig, axes = plt.subplots(n_var-1,1,figsize=[5,n_var*1.55])

    
    plt.subplots_adjust(hspace = .05, right = 0.8, bottom = 0.2)

    fig.suptitle(sup_title,fontsize =20 )
    
    for i ,varr in enumerate(DF.columns):
        plot_var(cfg,DF.copy(), i,varr, sup_title,y_label,axes)
    
    save_dir = (cfg['home_dir'] + cfg['fig_dir'])    
    fig.savefig(save_dir + sup_title+".png")
    return fig, axes
    

def plot_var(cfg,DF_plot, i,varr, sup_title,x_label,axes):
    DF_plot['index'] = DF_plot.index
    color = cfg['colors'][varr]
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
        
def plot_results(cfg,df, df1, df_experiment):    
    fig = 0
    axes = 0
    
    fig, axes = plot_each(cfg,df_experiment,'Results over episodes','episode',fig=fig, axes =axes)
    
    fig, axes = plot_each(cfg,df1,'First episode','step',fig=fig, axes =axes)
    
    fig, axes = plot_each(cfg,df,'Last episode','step',fig=fig, axes =axes)
      


