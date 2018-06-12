#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:53:40 2018

@author: ninalopatina
"""

def importer():
    #set running params
    plotting2d = 1
    plotting3d = 0
    RL = 0
    
    #files & folders
    home_dir = '/Users/ninalopatina/Desktop/Rocket_RL/'
    code_dir = 'code/'
    data_dir = 'data/'
    fig_dir = 'figures/'
#    data_file = 'more_data2.csv' #most of the data
    data_file = 'data1.csv' #initial data set
    
    #naming
    new_columns = ['I_CH4_g/s','I_CH4_t','I_O2_g/s','I_O2_t','O_CH4_flow_uniformity','O_CH4_mol_frac','O_T']
    in_var = new_columns[:4]
    out_var = new_columns[4:]
    #corresponding names, for figure labeling: p
    in_var_name = ['CH4 flow rate, g/s', 'CH4 temp (K)', 'O2 flow rate, g/s','O2 temp (K)'] #
    out_var_name = ['CH4 flow uniformity', 'CH4 mol frac', 'ouput temp (K)']
    
    #data
        
    #Observations that are the basis of plotted variables: 
    #O2 flow rate is highly predictive of CH4 flow uniformity & CH4 mol frac
    #O2 temp is highly predictive of output temp
    #CH4 flow rate is slightly predictive of CH4 mol frac
    #O2 temp is slightly predictive of CH4 flow uniformity
    
    #Plot the specific variables based on the above observations:
    xs = ['I_O2_t', 'I_O2_g/s','I_O2_g/s','I_CH4_g/s','I_O2_t']
    ys = ['O_T', 'O_CH4_flow_uniformity','O_CH4_mol_frac','O_CH4_mol_frac','O_CH4_flow_uniformity'] 
    x_names = ['O2 temp (K)','O2 flow rate, g/s','O2 flow rate, g/s','CH4 flow rate, g/s','O2 temp (K)'] #corresponding name
    y_names = ['ouput temp (K)','CH4 flow uniformity','CH4 mol frac','CH4 mol frac','CH4 flow uniformity'] #corresponding name
    
  
    class Bunch(object):
            def __init__(self, adict):
                self.__dict__.update(adict)
    
    params = Bunch(locals())
    
    #    return num_sess,max_trials,colnames,colsum, rnames, cols,ymin, ymax_ch , ymax_cen,ymax_cut,ymaxITI,    yminITI, gs_corr, colors, meanings,figsize ,gridspecsR ,gridspecsC ,title_size,table_font ,    label_size,hscale ,vscale ,top_dist,mat_convert ,make_figs,run_subs ,save_nb, pickled, drive_path_data, drive_path_code, subs
    return params
