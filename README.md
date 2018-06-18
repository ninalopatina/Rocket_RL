# Rocket_RL
###Reinforcement learning agent that runs fluid dynamics simulations for rocket engines

##By: Nina Lopatina
### This project was developed when I was an Insight AI Fellow, summer 2018. 

##About
This package presently imports fluid dynamics simulation data and plots the best-defined features. 

## Contents
config.config.yml has all of the user-set values and is called by the main function
RocketRL.python.main.py is the main function that calls the other scripts and data.
RocketRL.python.func.data_processing.py has all of the data importing, processing, and visualization functions
RocketRL.python.func.run_env.py runs the agent in the custom environment
RocketRL.python.envs.my_collection.my_awesome_env.py contains the custom env for this task

##To run:
1. Install requirements from req.txt; if you're using conda, create your new environment as below: 
'''
$ conda create -n newenvironment --file req.txt
'''

2. Set home_dir in main.py & config.yml 

3. Copy files that are in the \envs folder into your gym/envs folder (with algorithmic, classic_control, etc. folders in it). The lines in \envs\__init__.py that were added to the pre-existing file are: 
'''
from gym.envs.my_collection.my_awesome_env import SimpleCorridor
register(
   	id='SimpleCorridor-v0',
   	entry_point='gym.envs.my_collection:SimpleCorridor',
)
'''

4. In terminal, cd to RocketRL.python folder, 
'''
source activate newenvironment
python main.py
'''