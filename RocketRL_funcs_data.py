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

#import the parameters from the parameters function
import RocketRL_params
params = RocketRL_params.importer()

#clear workspace
plt.close('all')

def load_data(data_file):
    #loads csv data    
    df = pd.read_csv(params.home_dir+params.data_dir+data_file)
    #renaming to easier names & changing kg to g for visualization purposes    
    df.columns = params.new_columns
    df[['I_CH4_g/s','I_O2_g/s']] = df[['I_CH4_g/s','I_O2_g/s']] *1000
    return df

def graph(formula, m,c, x_range):  
    #graph the line over linear data
    x = np.array(x_range)  
    y = formula(x,m,c)  # <- note now we're calling the function 'formula' with x
    plt.plot(x, y, c='red')  
    
def my_formula(x,m,c):
    return m*x +c

def plot_3var(df):
    #plots 3 input variables in 3d, with output variable as the color of the points
    #for 3 output variables, this will create 3 plots.
    #user manually adjusts which 3/4 input variables to plot
    #to be able to move this around in Spyder, change your preferences to:
    #IPython console --> Graphics --> Backend: set to Automatic. 
    for var_o in params.out_var: 
        threedee = plt.figure().gca(projection='3d')
        #manually set which 3 out of 4 variables you want to visualize in a 3d plot:
        x = params.in_var[0]
        y = params.in_var[1]
        z = params.in_var[2] 
        p = threedee.scatter(xs = df[x], ys = df[y], zs= df[z] ,c=df[var_o])
        threedee.set_xlabel(x), threedee.set_ylabel(y), threedee.set_zlabel(z)
        threedee.set_title(var_o) #title is the output variables 
        plt.colorbar(p)
        plt.show()

def plot_2var(df,x,y,i):
    #this function saves a png of the plotted variables 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df.plot.scatter(x, y, ax = ax )
    xx = params.x_names[i] #corresponding name
    yy = params.y_names[i] #corresponding name
    ax.set_xlabel(xx)
    ax.set_ylabel(yy)
    ax.set_title(yy + ' by ' + xx)
    if i == 0: #first x & y I'm plotting have a linear relationship
        #get the r-squared value: 
        model = sm.OLS(df[y], df[x]).fit()
#        predictions = model.predict(df[x]) # make the predictions by the model
        #label the r-squared value: 
        s = ('r-squared = ' + str(round(model.rsquared,3)))
        ax.annotate(s,(450,250),size= 16)
        #get the equation of the line: 
        A = vstack([df[x].values,ones(len(df[x]))]).T
        m, c = lstsq(A, df[y].values)[0]
        #label the equation of the line: 
        s2 = ("Line Solution is y = {m}x + {c}".format(m=round(m,2),c=round(c,2)))
        ax.annotate(s2,(350,200),size= 12)
        #draw the line:
        graph(my_formula,m,c, x_range=range(200, 1000))
        # Print out the statistics
        print(model.summary())
    #save each as png    
    save_dir = (params.home_dir + params.fig_dir)    
    fig.savefig(save_dir + str(i)+".png")
        
def data_process():
    df = load_data(params.data_file)
    
    #visualize the data in 3d:
    if params.plotting3d == 1:
        plot_3var(df)
        
    if params.plotting2d == 1: #plot each pair of input/output variables you've chosen
        for i, x in enumerate(params.xs): 
            y = params.ys[i]
            plot_2var(df,x,y,i)