#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:57:40 2018

@author: ninalopatina
"""
#import all the packages for funcs
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from numpy import ones,vstack
from numpy.linalg import lstsq
from matplotlib.backends.backend_pdf import PdfPages
import gym

#class RocketData():

#clear workspace
plt.close('all')

def plotter(plotd):
    if '2d' in plotd:
#        self.plotting2D = True
        plotting2D = True
    if '3d' in plotd:
#        self.plotting3D = True
        plotting3D = True
  

def load_data(cfg,data_file):
    #loads csv data    
    df = pd.read_csv(cfg['home_dir']+cfg['data_dir']+data_file)
    
    #renaming to easier names & changing kg to g for visualization purposes            
    return df

def clean_data(cfg,df):
    df.columns = cfg['all_var'] #change column names to what's in config.yaml
    df[cfg['change_scale']] = df[cfg['change_scale']] *1000 #scale these for visualization
    return df

def graph(formula, m,c, x_range):  
    #graph the line over linear data
    x = np.array(x_range)  
    y = formula(x,m,c)  # <- note now we're calling the function 'formula' with x
    plt.plot(x, y, c='red')  
    
def my_formula(x,m,c):
    return m*x +c

def plot_3var(cfg,df):
    #plots 3 input variables in 3d, with output variable as the color of the points
    #for 3 output variables, this will create 3 plots.
    #user manually adjusts which 3/4 input variables to plot
    #to be able to move this around in Spyder, change your preferences to:
    #IPython console --> Graphics --> Backend: set to Automatic. 
    for var_o in cfg['out_var']: 
        threedee = plt.figure().gca(projection='3d')
        #manually set which 3 out of 4 variables you want to visualize in a 3d plot:
        x = cfg['in_var'][1]
        y = cfg['in_var'][2]
        z = cfg['in_var'][0]
        p = threedee.scatter(xs = df[x], ys = df[y], zs= df[z] ,c=df[var_o])
        threedee.set_xlabel(x), threedee.set_ylabel(y), threedee.set_zlabel(z)
        threedee.set_title(var_o) #title is the output variables 
        plt.colorbar(p)
#        plt.show()
        save_dir = (cfg['home_dir'] + cfg['fig_dir'])    
        plt.savefig(save_dir + var_o+".png")

def plot_2var(cfg,df,x,y,i):
    #this function saves a png of the plotted variables 
#    df['I_O2_t'] = df['I_O2_t']/10
#    df['O_t'] = df['O_t']/10
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df.plot.scatter(x, y, ax = ax )
    xx = cfg['x_names'][i] #corresponding name
    yy = cfg['y_names'][i] #corresponding name
    ax.set_xlabel(xx)
    ax.set_ylabel(yy)
    ax.set_title(yy + ' by ' + xx)
    if i == 0: #first x & y I'm plotting have a linear relationship
        #get the r-squared value: 
        model = sm.OLS(df[y], df[x]).fit()
#        predictions = model.predict(df[x]) # make the predictions by the model
        #label the r-squared value: 
        s = ('r-squared = ' + str(round(model.rsquared,3)))
        ax.annotate(s,(450,350),size= 16)
        #get the equation of the line: 
        A = vstack([df[x].values,ones(len(df[x]))]).T
        m, c = lstsq(A, df[y].values)[0]
        #label the equation of the line: 
        s2 = ("Line Solution is y = {m}x + {c}".format(m=round(m,2),c=round(c,2)))
        ax.annotate(s2,(350,250),size= 12)
        #draw the line:
        graph(my_formula,m,c, x_range=range(200, 1000))
        # Print out the statistics
        print(model.summary())
    #save each as png    
    save_dir = (cfg['home_dir'] + cfg['fig_dir'])    
    fig.savefig(save_dir + str(i)+".png")
 
def make_mini(cfg,df):
    approx = df.round(0)
    approx['I_CH4_t'] = round(approx['I_CH4_t']/10,0)*10
    approx['I_O2_t'] = round(approx['I_O2_t']/10,0)*10
    mini_df = approx.copy()



    for var in cfg['in_var'][:-1]:
        mini_df = mini_df[mini_df[var]== float(mini_df[var].mode())]  

    print(mini_df.size)
#    mini_df.plot.scatter(x='I_O2_t', y = 'O_T')
    return mini_df
       
def data_process(cfg,plotting2d):
    df = load_data(cfg,cfg['data_file'])
    df = clean_data(cfg,df)
    #visualize the data in 3d:
#    if self.plotting3d == True:
#        plot_3var(df)
        
#    if self.plotting2d == True: #plot each pair of input/output variables you've chosen
    
    plot_3var(cfg,df)
    
    df_mini = make_mini(cfg, df)
    
    if plotting2d == True:
        for i, x in enumerate(cfg['xs']): 
            y = cfg['ys'][i]
            plot_2var(cfg,df_mini,x,y,i)
            
    return df, df_mini