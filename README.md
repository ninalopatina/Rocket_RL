# Rocket_RL
### Reinforcement learning agent that runs fluid dynamics simulations for rocket engines.

## About
This project was developed when I was an Insight AI Fellow, summer 2018. 
This package presently imports fluid dynamics simulation data and plots the best-defined features. 

## Contents
* config.data.yml has all of the user-set values for the data functions is called by the main function.
* config.model.yml has all of the user-set values for the model functions is called by the main function.
* RocketRL.python.main.py is the main function that calls the other scripts and data.
* RocketRL.python.func.data_processing.py has all of the data importing, processing, and visualization functions.
* RocketRL.python.func.run_env.py runs the agent in the custom environment.
* RocketRL.python.envs.my_collection.my_awesome_env.py contains the custom env for this task.

## To run:
1. This only works for OSX: Install requirements from req.txt; if you're using conda, create your new environment as below: 

```Terminal
$ conda create -n newenvironment --file req.txt
```

2. In the root of your github repo, type :
```Bash
export PYTHONPATH=$PYTHONPATH:`pwd`/RocketRL/python
```

3. Copy files that are in the \envs folder into your gym/envs folder (with algorithmic, classic_control, etc. folders in it). The lines in \envs\__init__.py that were added to the pre-existing file are: 

```python 
from gym.envs.my_collection.my_awesome_env import SimpleCorridor
register(
   	id='SimpleCorridor-v0',
   	entry_point='gym.envs.my_collection:SimpleCorridor',
)
```

4. In terminal,
```Terminal
source activate newenvironment
python RocketRL/python/main.py
```

## To customize:
In the config.model.yml file, there are a few variables you can play with to get a feel for the model:
* n_exp changes how many episodes the model runs for
* gamma is the forgetting rate: this scales the relevancy of past rewards for the Q-value updates
