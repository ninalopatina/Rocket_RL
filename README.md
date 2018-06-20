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

(note to self: update req.txt)
1. This only works for OSX: Install requirements from req.txt; if you're using conda, create your new environment as below: 

```bash
$ conda create -n newenvironment --file req.txt
```
2. Install Ray: 
- A few notes: Download the latest tag instead of the main branch b/c main branch doesn't build
```bash
git clone https://github.com/ray-project/ray.git
cd ray
git checkout tags/ray-0.4.0
```
- (Install instructions)[http://ray.readthedocs.io/en/latest/installation.html] 

3. Install RLlib:
- install snappy first if you don't already have it:
```bash
$ brew install snappy
```
-(Install instructions)[https://ray.readthedocs.io/en/latest/rllib.html]

4. In the root of your github repo, type :
```Bash
export PYTHONPATH=$PYTHONPATH:`pwd`/RocketRL/python
```

5. In terminal,
```Bash
source activate newenvironment
python RocketRL/python/main.py
```

## To customize:
In the config.model.yml file, there are a few variables you can play with to get a feel for the model:
* n_exp changes how many episodes the model runs for
* gamma is the forgetting rate: this scales the relevancy of past rewards for the Q-value updates
