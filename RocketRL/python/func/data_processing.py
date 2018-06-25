#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:57:40 2018

@author: ninalopatina
"""
#import all the packages for funcs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from numpy import ones,vstack
from numpy.linalg import lstsq
import pickle
import yaml
import os

# Import function to create training and test set splits
from sklearn.cross_validation import train_test_split
# Import function to automatically create polynomial features! 
from sklearn.preprocessing import PolynomialFeatures
# Import Linear Regression and a regularized regression function
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
# Finally, import function to make a machine learning pipeline
from sklearn.pipeline import make_pipeline

#set config path
config_dir = 'config/'
CWD_PATH = os.getcwd()
config_path = os.path.join(CWD_PATH,config_dir,"data.yml")

with open(config_path, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

#TO DO: is below kosher?

all_var = cfg['in_var'].copy()
for item in cfg['out_var']:
    all_var.append(item)
cfg['all_var']= all_var


#class RocketData():

#clear workspace
plt.close('all')

def load_data(data_path):
    #loads csv data
    df = pd.read_csv(data_path)
    return df

def clean_data(cfg,df):
    #renaming to easier names & changing kg to g for visualization purposes
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

def save_plot(fig,title):
    save_dir = os.path.join(CWD_PATH, cfg['result_path'],cfg['fig_path'])
    fig.savefig(save_dir + title+".png")

def plot_3dvar(cfg,df):
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
        save_plot(plt,title = var_o)

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
    #save each as png
    save_plot(fig,str(i))


def plot_3var(cfg,df,x,y,i):
    #this function saves a png of the plotted variables 
#    df['I_O2_t'] = df['I_O2_t']/10
#    df['O_t'] = df['O_t']/10
    
    for cvar in cfg['out_var']:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        df.plot.scatter(x, y,c=cvar, ax = ax )
        save_plot(fig,(str(i)+cvar))
#    xx = cfg['x_names'][i] #corresponding name
#    yy = cfg['y_names'][i] #corresponding name
#    ax.set_xlabel(xx)
#    ax.set_ylabel(yy)
#    ax.set_title(yy + ' by ' + xx)
#    if i == 0: #first x & y I'm plotting have a linear relationship
#        #get the r-squared value: 
#        model = sm.OLS(df[y], df[x]).fit()
##        predictions = model.predict(df[x]) # make the predictions by the model
#        #label the r-squared value: 
#        s = ('r-squared = ' + str(round(model.rsquared,3)))
#        ax.annotate(s,(450,350),size= 16)
#        #get the equation of the line: 
#        A = vstack([df[x].values,ones(len(df[x]))]).T
#        m, c = lstsq(A, df[y].values)[0]
#        #label the equation of the line: 
#        s2 = ("Line Solution is y = {m}x + {c}".format(m=round(m,2),c=round(c,2)))
#        ax.annotate(s2,(350,250),size= 12)
#        #draw the line:
#        graph(my_formula,m,c, x_range=range(200, 1000))
    #save each as png    


def make_round(cfg,df):
    approx = df.round(1)
    approx['I_CH4_t'] = round(approx['I_CH4_t']/10,0)*10
    approx['I_O2_t'] = round(approx['I_O2_t']/10,0)*10
    return approx

def make_mini(cfg,df):
    mini_df = df.copy()
    for var in cfg['in_var'][:-1]:
        mini_df = mini_df[mini_df[var]== float(mini_df[var].mode())]
    return mini_df

def make_mini2(cfg,df):
    i = 0
    my_var = cfg['in_var']
    for var in my_var:
        #pick a 2nd var:
        my_var2 = my_var.copy()
        my_var2.remove(var)
        #for the other 3 var
        for var2 in my_var2:
            mini_df = df.copy()
            my_var3 = my_var2.copy()
            my_var3.remove(var2)
            for var3 in my_var3:
                mini_df = mini_df[mini_df[var3]== float(mini_df[var3].mode()[0])]

            i = i + 1
            plot_3var(cfg,mini_df,var,var2,i)

def data_process(plot_data,run_regression,save_regression):
    data_path = os.path.join(CWD_PATH,cfg['data_file_path'])
    df = load_data(data_path)
    df = clean_data(cfg,df)
    
    if (plot_data==1) | (plot_data==3):
        plot_3dvar(cfg,df)
 
    approx = make_round(cfg,df)
    

    if (plot_data==1) | (plot_data==2):
        df_mini = make_mini2(cfg,approx)
        
        df_mini = make_mini(cfg, approx)
        for i, x in enumerate(cfg['xs'][:1]):
            y = cfg['ys'][i]
            plot_2var(cfg,df_mini,x,y,i)

    if run_regression == True:
        reg_runner(df,save_regression)



    return df, approx


def linspacer(df,num_items,col):
    out = np.linspace(df.min()[df.columns[col]],df.max()[df.columns[col]],num_items)
    return out

def reg_runner(df,save_regression):
    #linear  regression
    df_inputs = df[df.columns[:4]]
    #blank dataframe for your predictions:
    df_pred = pd.DataFrame(columns = df_inputs.columns)
    
    for var in cfg['out_var']:
    
        X_train, X_test, y_train, y_test = train_test_split(df_inputs, 
                            df[var],test_size=cfg['test_set_fraction'])
        
        # Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
        for degree in range(cfg['degree_min'],cfg['degree_max']+1):
            model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), 
                    LassoCV(n_alphas=cfg['lasso_nalpha'],eps=cfg['lasso_eps'],
                            max_iter=cfg['lasso_iter'],normalize=True,cv=cfg['cv']))
            model.fit(X_train,y_train)
            test_pred = np.array(model.predict(X_test))
            RMSE=np.sqrt(np.sum(np.square(test_pred-y_test)))
            test_score = model.score(X_test,y_test)
            print('Regression results for',var)
            print(var,test_score)
        
        df_new = pd.DataFrame(columns = df_inputs.columns)
        
        #now make a interpolated dataframe: 
        num_items = 50
        w = linspacer(df_inputs,num_items,col=0)        
        x = linspacer(df_inputs,num_items,col=1)
        y = linspacer(df_inputs,num_items,col=2)
        z = linspacer(df_inputs,num_items,col=3)
        wv, xv, yv, zv = np.meshgrid(w,x, y,z)
        
        for name, values in zip(df_inputs.columns,[wv,xv,yv,zv]):
            df_new[name] = pd.Series(values.flatten())
            
        df_pred[var] = pd.Series(model.predict(df_new))
          
    df_pred[cfg['in_var']] = df_new
    
    df_pred.plot(subplots=True)
    
    if save_regression == True:
        pickle_path = os.path.join(CWD_PATH,cfg['result_path'],cfg['pickle_path'],"reg.p")
        df_pred.to_pickle(pickle_path)
    
    
    
    
    #TO DO: set this up in class
#def plotter(plotd):
#    if '2d' in plotd:
##        self.plotting2D = True
#        plotting2D = True
#    if '3d' in plotd:
##        self.plotting3D = True
#        plotting3D = True
