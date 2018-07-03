"""This is the custom OpenAI environment for rocket engine tuning"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym.spaces import Box, Tuple
import random

import pickle
import os
import yaml


CWD_PATH = os.getcwd()
    
class AllVar(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50}

    def __init__(self,config = None):
        self.repo_dir = 'gitrepos/Rocket_RL/'
        self.label_path = 'video_labels/'
        self.pick_path = 'results/pickles/' 

        # User set values are below: 
        self.num_ins = 4
        
        self.MSE_thresh1 = 10
        self.MSE_thresh2 = 10
        self.MSE_thresh3 = 10
        
        self.rew_goal = 10000
        self.action_range = 20
        
        self.noise = 0
        self.minmaxbuffer = 0

        # Get the function of input-output values:
        self.O_CH4_flow_uniformity= self.get_funcs('O_CH4_flow_uniformity')
        self.O_CH4_mol_frac = self.get_funcs('O_CH4_mol_frac')
        self.O_t = self.get_funcs('O_t')
        
        #get maxes & mins from any file
        [maxes,mins] = self.get_max_min('O_CH4_flow_uniformity')
        
        # For rendering:
        self.viewer = None
        self.labels = ['G1F','G1T','G2F','G2T','O1','O2','O3']
          
        # For ref, these are:
        #in: 1 ch4 flow, 2 ch4 t, 3 o2 flow, 4 o2 t,
        #out: 5 flow unif, 6 mol frac, 7 temp

        # Feature engineering so your data lines up on order of magnitude:
        self.scaler = [40, .2, 20, .1, 100, 100, .1]
        self.undo_scaler = np.divide(1,self.scaler)

        self.mins = mins * self.scaler
        self.maxes = maxes * self.scaler

        self.action_space = Box(-self.action_range, self.action_range, shape=(self.num_ins,), dtype=np.float32)

        self.observation_space = Tuple((Box(self.mins.values[0],self.maxes.values[0],shape=(1,), dtype=np.float32),
                                               Box(self.mins.values[1],self.maxes.values[1],shape=(1,), dtype=np.float32),
                                               Box(self.mins.values[2],self.maxes.values[2],shape=(1,), dtype=np.float32),
                                               Box(self.mins.values[3],self.maxes.values[3],shape=(1,), dtype=np.float32),
                                               Box(self.mins.values[4],self.maxes.values[4],shape=(1,), dtype=np.float32),
                                               Box(self.mins.values[5],self.maxes.values[5],shape=(1,), dtype=np.float32),
                                               Box(self.mins.values[6],self.maxes.values[6],shape=(1,), dtype=np.float32),
                                               Box(self.mins.values[4],self.maxes.values[4],shape=(1,), dtype=np.float32),
                                               Box(self.mins.values[5],self.maxes.values[5],shape=(1,), dtype=np.float32),
                                               Box(self.mins.values[6],self.maxes.values[6],shape=(1,), dtype=np.float32)))

        # TODO this isn't really a proper gym spec
        self._spec = lambda: None
        self._spec.id = "AllVar-v0"
        self.episode = 0
        self.reward = 0
        self.reset()

    def get_funcs(self,var):
        fname = (var+".p")      
        try:
            pickle_path = os.path.join(CWD_PATH,self.repo_dir,self.pick_path,fname) #on the second run
            [coef,powers,intercept,maxes,mins] = pickle.load(open(pickle_path,'rb'))
            self.label_dir = os.path.join(CWD_PATH, self.repo_dir,self.label_path)
        
        except Exception as e:
            print('error: %s'%(e))
            pickle_path = os.path.join(CWD_PATH,'../../..',self.repo_dir,self.pick_path,fname) #on the first run
            [coef,powers,intercept,maxes,mins] = pickle.load(open(pickle_path,'rb'))
            self.label_dir = os.path.join(CWD_PATH,'../../..', self.repo_dir,self.label_path)

        
        #the 3 function variables you need to-recreate this model
        #also the min & max to set this in the environment

        out = {'coef': coef, 'powers':powers,'intercept':intercept}
        return out

    def get_max_min(self,var):
        fname = (var+".p")
        try:
            pickle_path = os.path.join(CWD_PATH,self.repo_dir,self.pick_path,fname)
            [coef,powers,intercept,maxes,mins] = pickle.load(open(pickle_path,'rb'))
            self.label_dir = os.path.join(CWD_PATH, self.repo_dir,self.label_path)
            
        except Exception as e:
            print('error: %s'%(e))
            pickle_path = os.path.join(CWD_PATH,'../../..',self.repo_dir,self.pick_path,fname)
            [coef,powers,intercept,maxes,mins] = pickle.load(open(pickle_path,'rb'))
            self.label_dir = os.path.join(CWD_PATH,'../../..', self.repo_dir,self.label_path)

        #the 3 function variables you need to-recreate this model
        #also the min & max to set this in the environment
        return maxes,mins

    def temp_func(self,var,spot):
        y = var['intercept']
        for p,c in zip(var['powers'],var['coef']):
            #exp the 4 inputs to the power for that coef

            #to plug them into the equation, un-scale them:
            a = ((self.ins*self.undo_scaler[:4])**p)
            y += c* np.prod(a)

        #to fit this into the environment, re-scale:
        y = y * self.scaler[spot]

        noise = random.randint(-1,1) * random.random() * self.noise #scales noise
        y += noise

        return y

    def test_viable(self,outs):
        viable = True
        for i,temp_i in enumerate(outs):
            if (temp_i <= self.mins[i+4]):
                viable = False
            elif (temp_i >= self.maxes[i+4]): 
                viable = False
        return viable
    
    def reset(self): #start over
        self.steps = 0
        #on every reset, you have a new & goal temp
        #after the first episode, you start from where you ended on the last episode

        if self.episode == 0:
            self.ins = random.uniform(self.mins.values[:4],self.maxes.values[:4])
            #get the corresponding outputs:
            out_flow = self.temp_func(var=self.O_CH4_flow_uniformity,spot=4)
            out_frac = self.temp_func(var=self.O_CH4_mol_frac,spot =5)
            out_temp = self.temp_func(var=self.O_t,spot=6)

            outs = np.array([out_flow,out_frac,out_temp])
            self.starts = np.append(self.ins, outs)

        else:
            self.starts = self.state[:7] #previous episode's end state

        #get goals from random inputs:
        viable = False
        while viable == False:
            self.ins = random.uniform((self.mins.values[:4]+(self.mins.values[:4]*self.minmaxbuffer)),self.maxes.values[:4]-(self.maxes.values[:4]*self.minmaxbuffer))
    
            out_flow = self.temp_func(var=self.O_CH4_flow_uniformity,spot=4)
            out_frac = self.temp_func(var=self.O_CH4_mol_frac,spot =5)
            out_temp = self.temp_func(var=self.O_t,spot=6)

            outs = np.array([out_flow,out_frac,out_temp])
            
            #check viable
            viable = self.test_viable(outs)


        self.goals = outs

        #these are your current inputs
        self.ins = self.starts[:4]
        #state carries the starting points and the goals
        self.state = np.append(self.starts,self.goals)

        self.episode += 1
        self.tot_rew = 0

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

        #check that this is a viable output; if not, reject the action
        #is this temp change viable?
    
        MSE1 = (self.goals[0]-out_flow)**2
        MSE2 = (self.goals[1]-out_frac)**2
        MSE3 = (self.goals[2]-out_temp)**2

        MSE = MSE1 +  MSE2 + MSE3

        #update your state
        #add other goals here
        state_new = np.append(self.ins,[out_flow,out_frac,out_temp] )
        self.state =np.append(state_new,self.goals)

        done = ((MSE1 <= self.MSE_thresh1) & (MSE2 <= self.MSE_thresh2) & (MSE3 <= self.MSE_thresh3))
        done = bool(done)

            #get the corresponding reward
        reward = 0
        if done:
            reward += self.rew_goal
        else:          
            reward -= MSE

        self.reward = reward
        self.tot_rew += reward
        self.done = done

        return (self.state, reward, done, {'MSE thresh': self.MSE_thresh1})

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 550

        n_sect = 7
        world_width = n_sect*2 #x axis is just pixels

        buff_axis = 15
        #bottom of the screen scales to the input/output range of values
        world_height_bottom = np.max(self.maxes)+buff_axis
        #top is an arbitary # to place objects
        world_height_top = 100

        world_top = .3 #top portion is separate
        world_bottom = 1-world_top

        screen_height_bottom = world_bottom*screen_height

        axes_line1 = screen_height*(world_bottom + .2)

        scalex = screen_width/world_width
        scaley_bottom= screen_height_bottom/world_height_bottom

        move_oval = -scalex*.2
        move_up= scaley_bottom * buff_axis*.5

        #set sizes of shapes:

        oval_length = 25.0
        oval_width = 50.0
        cartwidth = 70.0
        cartheight1 = 5.0 #goal
        cartheight2 = 5.0 #agent

        #plots:
        scalestep = 100 #number steps
        scalestep = screen_width/scalestep

        light_col = .7
        dark_col = 1
        c1 = .6
        c2 = .8
        c3 = 1
#
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #input states:

            #the temp agents
            temp1 = rendering.make_capsule(oval_length, oval_width)
            self.temptrans1 = rendering.Transform()
            temp1.add_attr(self.temptrans1)
            temp1.set_color(0,0,light_col)
            self.viewer.add_geom(temp1)

            temp2 = rendering.make_capsule(oval_length, oval_width)
            self.temptrans2 = rendering.Transform()
            temp2.add_attr(self.temptrans2)
            temp2.set_color(0,0,dark_col)
            self.viewer.add_geom(temp2)

            #flow agents:
            flow1 = rendering.make_capsule(oval_length, oval_width)
            self.flowtrans1 = rendering.Transform()
            flow1.add_attr(self.flowtrans1)
            flow1.set_color(light_col,0,light_col)
            self.viewer.add_geom(flow1)

            flow2 = rendering.make_capsule(oval_length, oval_width)
            self.flowtrans2 = rendering.Transform()
            flow2.add_attr(self.flowtrans2)
            flow2.set_color(dark_col,0,dark_col)
            self.viewer.add_geom(flow2)

            #ouput states:
            #out1:
            #the guage is a rectangle
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight2/2, -cartheight2/2
            guage1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.tempguage1 = rendering.Transform()
            guage1.add_attr(self.tempguage1)
            guage1.set_color(0,c3,0)
            self.viewer.add_geom(guage1)

            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight1/2, -cartheight1/2
            goal1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.tempgoal1 = rendering.Transform()
            goal1.add_attr(self.tempgoal1)
            goal1.set_color(c3,0,0)
            self.viewer.add_geom(goal1)

            #out2:
            #the guage1 is a rectangle
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight2/2, -cartheight2/2
            guage2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.tempguage2 = rendering.Transform()
            guage2.add_attr(self.tempguage2)
            guage2.set_color(0,c2,0)
            self.viewer.add_geom(guage2)

            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight1/2, -cartheight1/2
            goal2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.tempgoal2 = rendering.Transform()
            goal2.add_attr(self.tempgoal2)
            goal2.set_color(c2,0,0)
            self.viewer.add_geom(goal2)

            #out3:
            #the guage2 is a rectangle
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight2/2, -cartheight2/2
            guage3 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.tempguage3 = rendering.Transform()
            guage3.add_attr(self.tempguage3)
            guage3.set_color(0,c1,0)
            self.viewer.add_geom(guage3)

            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight1/2, -cartheight1/2
            goal3 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.tempgoal3 = rendering.Transform()
            goal3.add_attr(self.tempgoal3)
            goal3.set_color(c1,0,0)
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

            #top measurements:
            #reward plot
            #line:
            track = rendering.Line((scalex*1.5,axes_line1), (screen_width-scalex*1,axes_line1))
            self.trackis = rendering.Transform()
            track.add_attr(self.trackis)
            track.set_color(0,0,0)
            self.viewer.add_geom(track)

            #dot plot
            dot = rendering.make_circle(oval_length)
            self.dottrans = rendering.Transform()
            dot.add_attr(self.dottrans)
            dot.set_color(0,0,0)
            self.viewer.add_geom(dot)

            #labels: #to do: get this to work

            num = 0
            label_buff_y = 1.07
            label_buff_x = .2
            img_scale = .5
            img_wid = 179 *img_scale
            img_height = 124 * img_scale

            for label in self.labels:
                pth = (self.label_dir+label+'.png')
                self.txt = rendering.Image(pth,img_wid,img_height)
                locx = (num*2)+1
                self.txtis = rendering.Transform(translation=(scalex*locx +locx* label_buff_x,world_bottom*screen_height*label_buff_y))
                self.txt.add_attr(self.txtis)
                self.viewer.add_geom(self.txt)
                num = num+1
#

            #step label
            pth = (self.label_dir+'Step.png')
            self.txt = rendering.Image(pth,img_wid,img_height)
            self.txtis = rendering.Transform(translation=(scalex*.5,axes_line1))
            self.txt.add_attr(self.txtis)
            self.viewer.add_geom(self.txt)
#


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
        self.dottrans.set_translation(scalex*1.5 + self.steps*scalestep, axes_line1)
        done_grow = .5*self.done
        self.dottrans.set_scale(1+done_grow,1+done_grow)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
