"""This code is adapted from rllib's example of a custom env"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import random
import pandas as pd
import pickle


import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env

import os
import yaml

# ##TO DO: get cfg in here more elegantly
#set config path

#TO DO: change this to get to this dir from the ray dir: 

CWD_PATH = '/Users/ninalopatina/gitrepos/Rocket_RL/'
config_path = os.path.join(CWD_PATH,"config/config.yml")



with open(config_path, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
    


#TO DO: figure out how to add this to github repo
#TO DO: make subclasses and move everything from the init and other func to the proper subclass

class TwoTemp(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 1}
        
    def __init__(self,config = None):
        #for rendering:
        self.viewer = None
        self.label_dir = os.path.join(CWD_PATH, cfg['labels_path'])
        #get your func of input-output values
        

        self.O_CH4_flow_uniformity= self.get_funcs('O_CH4_flow_uniformity')
        self.O_CH4_mol_frac = self.get_funcs('O_CH4_mol_frac')
        self.O_t = self.get_funcs('O_t')
        
        #get maxes & mins from any file
        [maxes,mins] = self.get_max_min('O_CH4_flow_uniformity')

        #to do: move these to cfg 
        self.num_ins = 4

        self.MSE_thresh1 = cfg['MSE_thresh1']
        self.MSE_thresh2 = cfg['MSE_thresh2']
        self.MSE_thresh3 = cfg['MSE_thresh3']
        
        
        self.action_range = cfg['action_range']
        
        
        #for ref, these are: 
        #in: 1 ch4 flow, 2 ch4 t, 3 o2 flow, 4 o2 t, 
        #out: 5 flow unif, 6 mol frac, 7 temp
        
        #feature engineering so your data lines up on order of magnitude:
#        self.scaler = [40, .2, 20, .1, 100, 100, .1]
        #scales your variables
        self.scaler = cfg['scaler']
#        self.scaler = [1, 1, 1, 1, 1, 1, 1]
        self.undo_scaler = np.divide(1,self.scaler)
        
        self.mins = mins * self.scaler
        self.maxes = maxes * self.scaler

        self.action_space = spaces.Box(-self.action_range, self.action_range, shape=(self.num_ins,), dtype=np.float32)

        self.observation_space = spaces.Tuple((spaces.Box(self.mins.values[0],self.maxes.values[0],shape=(1,), dtype=np.float32), 
                                               spaces.Box(self.mins.values[1],self.maxes.values[1],shape=(1,), dtype=np.float32), 
                                               spaces.Box(self.mins.values[2],self.maxes.values[2],shape=(1,), dtype=np.float32),
                                               spaces.Box(self.mins.values[3],self.maxes.values[3],shape=(1,), dtype=np.float32),
                                               spaces.Box(self.mins.values[4],self.maxes.values[4],shape=(1,), dtype=np.float32), 
                                               spaces.Box(self.mins.values[5],self.maxes.values[5],shape=(1,), dtype=np.float32),
                                               spaces.Box(self.mins.values[6],self.maxes.values[6],shape=(1,), dtype=np.float32), 
                                               spaces.Box(self.mins.values[4],self.maxes.values[4],shape=(1,), dtype=np.float32),
                                               spaces.Box(self.mins.values[5],self.maxes.values[5],shape=(1,), dtype=np.float32),
                                               spaces.Box(self.mins.values[6],self.maxes.values[6],shape=(1,), dtype=np.float32)))
            
        # TODO this isn't really a proper gym spec
        self._spec = lambda: None
        self._spec.id = "TwoTemp-v0"
        self.episode = 0
        self.reset()
    
    def get_funcs(self,var):
        fname = (var+".p")
        pickle_path = os.path.join(CWD_PATH,cfg['result_path'],cfg['pickle_path'],fname)
        #the 3 function variables you need to-recreate this model
        #also the min & max to set this in the environment
        [coef,powers,intercept,maxes,mins] = pickle.load(open(pickle_path,'rb'))
        out = {'coef': coef, 'powers':powers,'intercept':intercept}

        return out
    
    
    def get_max_min(self,var):
        fname = (var+".p")
        pickle_path = os.path.join(CWD_PATH,cfg['result_path'],cfg['pickle_path'],fname)
        #the 3 function variables you need to-recreate this model
        #also the min & max to set this in the environment
        [coef,powers,intercept,maxes,mins] = pickle.load(open(pickle_path,'rb'))
        return maxes,mins
    
    def temp_func(self,var,spot):
        
        y = var['intercept']
        
        for p,c in zip(var['powers'],var['coef']):
            #exp the 4 inputs to the power for that coef
            
            #to plug them into the equation, un-scale them: 
            a = ((self.ins*self.undo_scaler[:4])**p)
            y += c* np.prod(a)
#        noise = random.randint(-1,1) * random.random() * .5 #from -.5 to .5
#        y += noise
            
        #to fit this into the environment, re-scale:
        y = y * self.scaler[spot]
        return y
        
#        old:
#        z1 = self.table.iloc[(self.table['I_CH4_t']-self.ins[0]*10).abs().argsort()[:50]]
#        lookup = z1.iloc[(z1['I_O2_t']-self.ins[1]*10).abs().argsort()[:1]].index
#        
#        z = z1.loc[lookup[0],'O_t']/10 #+noise
#        return z

    #to do: figure out how to put below into each subclass
    def reset(self): #start over 

        self.steps = 0
        if cfg['training'] == True:
            
            
            #on every reset, you have a new start & goal temp:
            self.starts = random.uniform(self.mins.values,self.maxes.values)
            if self.episode > 0:
                self.starts[4:] = self.goals #previous episode's goals
              
            self.ins = random.uniform((self.mins.values[:4]+(self.mins.values[:4]*cfg['minmaxbuffer'])),self.maxes.values[:4]-(self.maxes.values[:4]*cfg['minmaxbuffer']))
            
            out_flow = self.temp_func(var=self.O_CH4_flow_uniformity,spot=4)
            out_frac = self.temp_func(var=self.O_CH4_mol_frac,spot =5)
            out_temp = self.temp_func(var=self.O_t,spot=6)
            
            self.goals = np.array([out_flow,out_frac,out_temp])
            
        else: #if you're testing on a particular value, put them here
            if cfg['testing']==1: #small range
                self.starts = random.uniform(self.mins.values,self.maxes.values)
                self.goals(random.uniform(self.mins.values[4:],self.maxes.values[4:]))
            elif cfg['testing']==2: #rand
                self.starts = random.uniform(self.mins.values,self.maxes.values)
                self.goals(random.uniform(self.mins.values[4:],self.maxes.values[4:]))
            elif cfg['testing']==3:    #wide range
                self.starts = random.uniform(self.mins.values,self.maxes.values)
                self.goals(random.uniform(self.mins.values[4:],self.maxes.values[4:]))
        self.ins = self.starts[:4]
        self.state = np.append(self.starts,self.goals)
        
        self.episode += 1
        
        return (self.state)
        
    def step(self, action):
        #TO DO: Move this
        self.steps += 1
        in_temp = self.state[:4]
                
        #increase or decrease the input temp
        new_temp = in_temp+ action #self.temp_knob*self.action_map[action]

        #is this temp change viable?
        for i,temp_i in enumerate(new_temp):
            if (temp_i <= self.mins[i]):
                new_temp[i] = self.mins[i]
#                in_temp = new_temp
            elif (temp_i >= self.maxes[i]): #TO DO: fix this -1 ; idk why i get error otherwise
                new_temp[i] = self.maxes[i]
#                in_temp = new_temp
#            else:
#                in_temp += self.temp_knob*cfg['action_map'][act]
                
        in_temp = new_temp

        #get the corresponding output temp:
        self.ins = in_temp
        
        #get all the new outputs:
        out_flow = self.temp_func(var=self.O_CH4_flow_uniformity,spot=4)
        out_frac = self.temp_func(var=self.O_CH4_mol_frac,spot =5)
        out_temp = self.temp_func(var=self.O_t,spot=6)

        #get the MSE for reward function
        
        #2 goals:
        

        MSE1 = (self.goals[0]-out_flow)**2
        MSE2 = (self.goals[1]-out_frac)**2
        MSE3 = (self.goals[2]-out_temp)**2
        
        MSE = MSE1*cfg['MSE1_scale'] + MSE2*cfg['MSE2_scale'] + MSE3*cfg['MSE3_scale']

        #update your state
        #add other goals here
        state_new = np.append(self.ins,[out_flow,out_frac,out_temp] )
        self.state =np.append(state_new,self.goals)
        
        done = ((MSE1 <= self.MSE_thresh1) & (MSE2 <= self.MSE_thresh2) & (MSE3 <= self.MSE_thresh3))
        done = bool(done)

        #get the corresponding reward
        reward = 0
        if done:
            reward += cfg['rew']

        else:
            reward -= MSE
            
        self.reward = reward
        self.done = done

        return (self.state, reward, done, {})
    
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 550
        

        n_sect = 7
        world_width = n_sect*2 #x axis is just pixels
        
        world_height_bottom = np.max(self.maxes)+100
        world_height_top = 100
        
        world_top = .3 #top portion is separate
        world_bottom = 1-world_top
        
        
        scalex = screen_width/world_width
        scaley_bottom= (world_bottom*screen_height)/world_height_bottom
        scaley_top= (world_top*screen_height)/world_height_top
         
        move_oval = -scalex*.3
        move_up= scaley_bottom *10

        #set sizes of shapes:
        
        oval_length = 25.0
        oval_width = 50.0
        cartwidth = 70.0
        cartheight1 = 5.0 #goal
        cartheight2 = 5.0 #agent
#
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #input states:
            
            #the temp agents
            temp1 = rendering.make_capsule(oval_length, oval_width)
            self.temptrans1 = rendering.Transform()
            temp1.add_attr(self.temptrans1)
            temp1.set_color(0,0,.3)
            self.viewer.add_geom(temp1)
                        
            temp2 = rendering.make_capsule(oval_length, oval_width)
            self.temptrans2 = rendering.Transform()
            temp2.add_attr(self.temptrans2)
            temp2.set_color(0,0,.6)
            self.viewer.add_geom(temp2)
            
            #flow agents:
            flow1 = rendering.make_capsule(oval_length, oval_width)
            self.flowtrans1 = rendering.Transform()
            flow1.add_attr(self.flowtrans1)
            flow1.set_color(.3,0,.3)
            self.viewer.add_geom(flow1)
                        
            flow2 = rendering.make_capsule(oval_length, oval_width)
            self.flowtrans2 = rendering.Transform()
            flow2.add_attr(self.flowtrans2)
            flow2.set_color(.6,0,.6)
            self.viewer.add_geom(flow2)
            
            #ouput states:
            #out1:
            #the guage is a rectangle
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight2/2, -cartheight2/2
            guage1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.tempguage1 = rendering.Transform()
            guage1.add_attr(self.tempguage1)
            guage1.set_color(0,.9,0)
            self.viewer.add_geom(guage1)
            
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight1/2, -cartheight1/2
            goal1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.tempgoal1 = rendering.Transform()
            goal1.add_attr(self.tempgoal1)
            goal1.set_color(.9,0,0)
            self.viewer.add_geom(goal1)
            
            #out2:
            #the guage1 is a rectangle
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight2/2, -cartheight2/2
            guage2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.tempguage2 = rendering.Transform()
            guage2.add_attr(self.tempguage2)
            guage2.set_color(0,.7,0)
            self.viewer.add_geom(guage2)
            
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight1/2, -cartheight1/2
            goal2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.tempgoal2 = rendering.Transform()
            goal2.add_attr(self.tempgoal2)
            goal2.set_color(.7,0,0)
            self.viewer.add_geom(goal2)
            
            #out3:
            #the guage2 is a rectangle
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight2/2, -cartheight2/2
            guage3 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.tempguage3 = rendering.Transform()
            guage3.add_attr(self.tempguage3)
            guage3.set_color(0,.5,0)
            self.viewer.add_geom(guage3)
            

            
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight1/2, -cartheight1/2
            goal3 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.tempgoal3 = rendering.Transform()
            goal3.add_attr(self.tempgoal3)
            goal3.set_color(.5,0,0)
            self.viewer.add_geom(goal3)
            
            #line dividing all sections
            for l in range(n_sect):
                track = rendering.Line((scalex*((l*2)+1),0), (scalex*((l*2)+1),screen_height*world_bottom))
                self.trackis = rendering.Transform()
                track.add_attr(self.trackis)
                track.set_color(0,0,0)
                self.viewer.add_geom(track)
                
            #line top:
            track = rendering.Line((0,world_bottom*screen_height), (screen_width,world_bottom*screen_height))
            self.trackis = rendering.Transform()
            track.add_attr(self.trackis)
            track.set_color(0,0,0)
            self.viewer.add_geom(track)
            
            #labels: #to do: get this to work
          
#            num = 0
#            for fname, label in cfg['labels1'].items():
#                labels = 'labels1'
#                if labels == 'labels1':
#                    lbllen = len(label)
#                else: 
#                    lbllen = 1
#                    label = str(label)
#                pth = (self.label_dir+fname+cfg['ftype'])
#
#                img_wid = lbllen*cfg['label_mult']
#                img_height = cfg['label_height']
#                self.txt = rendering.Image(pth,img_wid,img_height)
#                locx = (num*2)+1
#                self.txtis = rendering.Transform(translation=(move_oval+scalex*locx,world_bottom*screen_height))
#                self.txt.add_attr(self.txtis)
#                self.viewer.add_geom(self.txt)
##
##                
#                num = num+1

        if self.state is None: return None

        x = self.state

        #4 ins:
        self.flowtrans1.set_translation(move_oval+scalex*1,move_up+scaley_bottom*x[0])
        self.temptrans1.set_translation(move_oval+scalex*3,move_up+scaley_bottom*x[1])
        self.flowtrans2.set_translation(move_oval+scalex*5,move_up+scaley_bottom*x[2])
        self.temptrans2.set_translation(move_oval+scalex*7,move_up+scaley_bottom*x[3])
        
        #3 outs:
        #flow
        self.tempguage1.set_translation(scalex*9,move_up+scaley_bottom*x[4])
        self.tempgoal1.set_translation(scalex*9,move_up+scaley_bottom*x[7])
        self.tempguage2.set_translation(scalex*11,move_up+scaley_bottom*x[5])
        self.tempgoal2.set_translation(scalex*11,move_up+scaley_bottom*x[8])
        self.tempguage3.set_translation(scalex*13,move_up+scaley_bottom*x[6])
        self.tempgoal3.set_translation(scalex*13,move_up+scaley_bottom*x[9])

        #top info:
      
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

