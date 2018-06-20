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
import yaml
import os

#set home directory
home_dir = '/Users/ninalopatina/gitrepos/Rocket_RL/'
config_dir = 'config/'

with open(home_dir + config_dir+"config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


#set main params;
plot_data = True
run_RL = False


#TO DO: is below kosher?

#add a few variables to cfg:
cfg['home_dir'] = home_dir

all_var = cfg['in_var'].copy()
for item in cfg['out_var']:
    all_var.append(item)
cfg['all_var']= all_var

#import the functions from the functions file
import func.data_processing as RocketData
import func.run_env as RocketRunEnv
import func.RL_results as RocketPlot

#TO do: set up class and run like this:
#rocketData = RocketData()
#RocketData.plotter('2d')

if plot_data == True:
    #Load and process data:
    df, approx = RocketData.data_process(cfg,plotting2d = False)

if run_RL == True:
    #Run your agent in the custom environment:
    df, df1, df_experiment = RocketRunEnv.run_temp(cfg)
    RocketPlot.plot_results(cfg,df, df1, df_experiment)


from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

##get a 2d model:
mini_df = approx.copy()
my_var = cfg['in_var']
my_var.remove('I_O2_t')
my_var.remove('I_CH4_t')
for var in my_var:
    mini_df = mini_df[mini_df[var]== float(mini_df[var].mode()[0])]

my_col = ['I_O2_t','I_CH4_t','O_t']
mini_df = mini_df[my_col]
#
#
data = mini_df.as_matrix()
data = data.transpose()
#

#figuring out how to interpolate my data here:

from scipy.interpolate import RegularGridInterpolator as rgi
my_interpolating_function = rgi(data)#, V)
Vi = my_interpolating_function(array([xi,yi,zi]).T)

##here we generate the new interpolated dataset,
##increase the resolution by increasing the spacing, 500 in this example
#new = interpolate.splev(np.linspace(0,1,500), tck, der=0)
#
##now lets plot it!
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.plot(data[0], data[1], data[2], label='originalpoints', lw =2, c='Dodgerblue')
#ax.plot(new[0], new[1], new[2], label='fit', lw =2, c='red')
#ax.legend()
#plt.savefig('junk.png')
#plt.show()
