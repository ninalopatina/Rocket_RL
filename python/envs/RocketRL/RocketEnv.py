"""This is a custom OpenAI environment for rocket engine tuning"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym.spaces import Box, Tuple
from gym.envs.classic_control import rendering
import random

import pickle
import os
import yaml

# Set config path. TO DO: import cfg from main script (from where RLlib calls this env)
CWD_PATH = os.getcwd()
cfg_path = 'config/config.yml'

#Figure out which directory you've called this env from:
try:
    join_path = 'Rocket_RL/'
    config_path = os.path.join(CWD_PATH,join_path,cfg_path)   
    
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
   
except Exception as e:
#    print('error: %s'%(e))
    join_path = '../../../Rocket_RL/'
    config_path = os.path.join(CWD_PATH,join_path,cfg_path)
   
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

 
class AllVar(gym.Env):
    """
    This class contains all of the functions for the custom Rocket Engine tuning environment. 
    """
    #For rendering the rollout:
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50}

    def __init__(self,config = None):
        """
        Variables can be set in config.yml
        """
        
        self.join_path = join_path
        self.label_path = cfg['labels_path']
        self.pick_path = (cfg['result_path'] + cfg['pickle_path'])
        self.label_dir = os.path.join(CWD_PATH,self.join_path, self.label_path)

        #Variables inherent to the Fluent data: 
        self.num_ins = 4

        self.scale_var = cfg['scale_var']
        # User set values are below. These can be adjusted in config.yml  
        self.MSE_thresh1 = (cfg['thresh1']*self.scale_var)**2
        self.MSE_thresh2 = (cfg['thresh2']*self.scale_var)**2
        self.MSE_thresh3 = (cfg['thresh3']*self.scale_var)**2
        
        self.rew_goal = cfg['reward'] * self.scale_var

        self.noise = cfg['noise']
        self.minmaxbuffer = cfg['minmaxbuffer']

        # Get the function of input-output mapping, and max & min:
        [self.O_CH4_flow_uniformity, mins,maxes] = self.get_funcs('O_CH4_flow_uniformity')
        [self.O_CH4_mol_frac,  mins,maxes] = self.get_funcs('O_CH4_mol_frac')
        [self.O_t, mins, maxes] = self.get_funcs('O_t')
        
        self.mins = mins# * self.scale_var
        self.maxes = maxes#* self.scale_var
        #Action range is a percentage of the total range
        self.action_range = cfg['action_range']*self.scale_var

        #Action space is the up & down range for the 4 actions 
        self.action_space = Box(-self.action_range, self.action_range, shape=(self.num_ins,), dtype=np.float32)

        # For ref, this is a 10d state space:
        #in: 1 ch4 flow, 2 ch4 t, 3 o2 flow, 4 o2 t,
        #out: 5 flow unif, 6 mol frac, 7 temp
        #out - target: 8 flow unif, 9 mol frac, 10 temp
        
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
        
        # For rendering:
        self.viewer = None
        self.labels = cfg['labels']
        
        #initialize variables for tracking:
        self.episode = 0
        self.reward = 0
        self.reset()

    def get_funcs(self,var):
        """
        This function loads the pickles with the function approximating the Fluent simulation data.
        """
        fname = (var+".p")         
        pickle_path = os.path.join(CWD_PATH,self.join_path,self.pick_path,fname)
        [coef,powers,intercept,mins,maxes] = pickle.load(open(pickle_path,'rb'))
        
        # The 3 function variables you need to-recreate this model & the min & max to set this in the environment.
        out = {'coef': coef, 'powers':powers,'intercept':intercept}
        return out, mins, maxes

    def temp_func(self,var):
        """
        This function is the observer in the RL model.
        The coef, powers, and intercept are used to create a function of the outputs given the inputs.
        There is an option to add noise, to approximate thermal noise or other fluctuations in the environment.
        """
        y = var['intercept']
        for p,c in zip(var['powers'],var['coef']):
            # Exp the 4 inputs to the power for that coef
            #to plug them into the equation, un-scale them:
#            a = np.multiply(self.ins,(1/self.scale_var))**p
            a = self.ins**p
            y += c* np.prod(a)

        #to fit this into the environment, re-scale:
        y = y #* self.scale_var

        # Noise is a random number (positive or negative), scaled by self.noise
        noise = random.randint(-1,1) * random.random() * self.noise * cfg['thresh1'] * self.scale_var #scales noise
        y += noise
        return y

    def test_viable(self,outs):
        """
        Because the regression model doensn't adhere to the bounds of the inputs, some of these outputs
        might be outside the range an engineer would encounter. This prevents such values from being set
        as targets, since that would be unrealistic.
        """
        
        viable = True
        for i,temp_i in enumerate(outs):
            if (temp_i <= self.mins[i+4]):
                viable = False
            elif (temp_i >= self.maxes[i+4]): 
                viable = False
        return viable
    
    def reset(self): 
        """ 
        This is the function to reset for every new episode. The starting position carries over from
        the previous episode. The goal temperature changes on every episode.
        """
        
        self.steps = 0
        if self.episode == 0:
            self.ins = random.uniform(self.mins.values[:4],self.maxes.values[:4])
            #get the corresponding outputs:
            out_flow = self.temp_func(var=self.O_CH4_flow_uniformity)
            out_frac = self.temp_func(var=self.O_CH4_mol_frac)
            out_temp = self.temp_func(var=self.O_t)

            outs = np.array([out_flow,out_frac,out_temp])
            self.starts = np.append(self.ins, outs)

        else:
            self.starts = self.state[:7] #previous episode's end state

        #get goals from random inputs:
        viable = False
        while viable == False:
            self.ins = random.uniform((self.mins.values[:4]+(self.mins.values[:4]*self.minmaxbuffer)),self.maxes.values[:4]-(self.maxes.values[:4]*self.minmaxbuffer))
            out_flow = self.temp_func(var=self.O_CH4_flow_uniformity)
            out_frac = self.temp_func(var=self.O_CH4_mol_frac)
            out_temp = self.temp_func(var=self.O_t)

            outs = np.array([out_flow,out_frac,out_temp])
            
            # Check if viable:
            viable = self.test_viable(outs)

        self.goals = outs

        # These are your current inputs:
        self.ins = self.starts[:4]
        # State carries the starting points and the goals.
        self.state = np.append(self.starts,self.goals)

        #Track episodes and total reward.
        self.episode += 1
        self.tot_rew = 0

        return (self.state)

    def step(self, action):
        """
        This function determines the outcome of every action.
        First, the env checks whether the action is within the min & max range of the inputs.
        Second, the corresponding output variables are calculated. 
        Third, the MSE is calculated. 
        The agent is done if the MSE is within the range specied in cfg, and rewarded accordingly.
        Otherwise, the agent is penalized by the amount of the MSE. 
        
        """
        self.steps += 1
        in_var = self.state[:4]

        # Increase or decrease the 4 input values
        new_var = in_var+ action 

        #If the agent tries to exceed the range of the mins & maxes, this sets them to the max. 
        for i,temp_i in enumerate(new_var):
            if (temp_i <= self.mins[i]):
                new_var[i] = self.mins[i]
            elif (temp_i >= self.maxes[i]): 
                new_var[i] = self.maxes[i]

        in_var = new_var

        # Get all the new outputs:
        self.ins = in_var
        out_flow = self.temp_func(var=self.O_CH4_flow_uniformity)
        out_frac = self.temp_func(var=self.O_CH4_mol_frac)
        out_temp = self.temp_func(var=self.O_t)

        #check that this is a viable output; if not, reject the action
        #is this temp change viable?
    
        MSE1 = (self.goals[0]-out_flow)**2
        MSE2 = (self.goals[1]-out_frac)**2
        MSE3 = (self.goals[2]-out_temp)**2

        MSE = MSE1 +  MSE2 + MSE3

        # Update your state:
        state_new = np.append(self.ins,[out_flow,out_frac,out_temp] )
        self.state =np.append(state_new,self.goals)

        done = ((MSE1 <= self.MSE_thresh1) & (MSE2 <= self.MSE_thresh2) & (MSE3 <= self.MSE_thresh3))
        done = bool(done)

        # Get the corresponding reward:
        reward = 0
        if done:
            reward += self.rew_goal
        else:          
            reward -= MSE *cfg['MSE_scale']

        self.reward = reward
        self.tot_rew += reward
        self.done = done

        return (self.state, reward, done, {'MSE thresh': self.MSE_thresh1})
    
    def make_oval(self,c1,c2,c3):
        temp = rendering.make_capsule(self.oval_length, self.oval_width)
        temptrans = rendering.Transform()
        temp.add_attr(temptrans)
        temp.set_color(c1,c2,c3)
        self.viewer.add_geom(temp)
        return temptrans
    
    def make_rect(self,c1,c2,c3):
        l,r,t,b = -self.rect_width/2, self.rect_width/2, self.rect_height/2, -self.rect_height/2
        gauge = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        outgauge = rendering.Transform()
        gauge.add_attr(outgauge)
        gauge.set_color(c1,c2,c3)
        self.viewer.add_geom(gauge)
        return outgauge
    
    def make_line(self,x1,y1,x2,y2):
        track = rendering.Line((x1,y1), (x2,y2))
        trackis = rendering.Transform()
        track.add_attr(trackis)
        track.set_color(0,0,0)
        self.viewer.add_geom(track)
        
    def render(self, mode='human'):
        """
        This function renders the agent's actions.
        The top of the screen tracks the # of steps
        The bottom of the screen is the inputs and outputs.
        The inputs are 4 ovals and the outputs are 3 rectangles.
        Red = goal position, and green = current position for the outputs. 
        """
        screen_width = 800
        screen_height = 550

        # Width is one column for each variable
        n_sect = 7
        world_width = n_sect*2 # X axis is just pixels
        
        buff_axis = cfg['buff_axis']
        #bottom of the screen scales to the input/output range of values
        world_height_bottom = np.max(self.maxes)+buff_axis
        
        # Top is for counting steps
        world_height_top = 100

        #Split the screen:
        world_top = .3
        world_bottom = 1-world_top
        screen_height_bottom = world_bottom*screen_height

        #Set where to draw the steps axis
        axes_line1 = screen_height*(world_bottom + .2)

        # Scale the pixels in the screen:
        scalex = screen_width/world_width
        scaley_bottom= screen_height_bottom/world_height_bottom

        # Some adjustments to move some objects up/ right
        move_oval = -scalex*.2
        move_up= scaley_bottom * buff_axis*.5

        #set sizes of shapes:
        self.oval_length = 25.0
        self.oval_width = 50.0
        self.rect_width = 70.0
        self.rect_height = 5.0 

        #Step plot:
        scalestep = screen_width/cfg['scalestep']

        #color shades:
        light_col = .7
        dark_col = 1
        c11 = .6
        c22 = .8
        c33 = 1

        if self.viewer is None:
            #TO DO: find an alternative to copy-paste to generate multiple similar shapes
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            #Input states:

            #the temp action
            self.temptrans1 = self.make_oval(0,0,light_col)
            self.temptrans2 = self.make_oval(0,0,dark_col)
            #flow action:
            self.flowtrans1 = self.make_oval(light_col,0,light_col)
            self.flowtrans2 = self.make_oval(dark_col,0,dark_col)

            #output states:
            #out1:
            #the gauge is a rectangle  
            self.outgauge1 = self.make_rect(0,c33,0)
            #goal is red rectangle
            self.outgoal1= self.make_rect(c33,0,0)
            
            #out2:
            #the gauge is a rectangle  
            self.outgauge2 = self.make_rect(0,c22,0)
            #goal is red rectangle
            self.outgoal2= self.make_rect(c22,0,0)

            #out3:
            #the gauge is a rectangle  
            self.outgauge3 = self.make_rect(0,c11,0)
            #goal is red rectangle
            self.outgoal3 = self.make_rect(c11,0,0)

            #lines on which "controls" sit
            for l in range(n_sect):   
                self.make_line(scalex*((l*2)+1),0, scalex*((l*2)+1),screen_height*world_bottom)

            # Line separating the top and bottom of the screen. 
            self.make_line(0,world_bottom*screen_height,screen_width,world_bottom*screen_height)
            # Step # axis.
            self.make_line(scalex*1.5,axes_line1,screen_width-scalex*1,axes_line1)

            # The dot tracking the step #
            dot = rendering.make_circle(self.oval_length)
            self.dottrans = rendering.Transform()
            dot.add_attr(self.dottrans)
            dot.set_color(0,0,0)
            self.viewer.add_geom(dot)

            #labels: 
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

            #step label
            pth = (self.label_dir+'Step.png')
            self.txt = rendering.Image(pth,img_wid,img_height)
            self.txtis = rendering.Transform(translation=(scalex*.5,axes_line1))
            self.txt.add_attr(self.txtis)
            self.viewer.add_geom(self.txt)

        if self.state is None: return None

        x = self.state

        # 4 ins:
        self.flowtrans1.set_translation(move_oval+scalex*1,move_up+scaley_bottom*x[0])
        self.temptrans1.set_translation(move_oval+scalex*3,move_up+scaley_bottom*x[1])
        self.flowtrans2.set_translation(move_oval+scalex*5,move_up+scaley_bottom*x[2])
        self.temptrans2.set_translation(move_oval+scalex*7,move_up+scaley_bottom*x[3])

        # 3 outs: current & goal:
        self.outgauge1.set_translation(scalex*9,move_up+scaley_bottom*x[4])
        self.outgoal1.set_translation(scalex*9,move_up+scaley_bottom*x[7])
        self.outgauge2.set_translation(scalex*11,move_up+scaley_bottom*x[5])
        self.outgoal2.set_translation(scalex*11,move_up+scaley_bottom*x[8])
        self.outgauge3.set_translation(scalex*13,move_up+scaley_bottom*x[6])
        self.outgoal3.set_translation(scalex*13,move_up+scaley_bottom*x[9])

        #step info:
        self.dottrans.set_translation(scalex*1.5 + self.steps*scalestep, axes_line1)
        done_grow = .5*self.done
        self.dottrans.set_scale(1+done_grow,1+done_grow) #expand size when done

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
