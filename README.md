# Rocket_RL
### Reinforcement learning agent that runs fluid dynamics simulations for rocket engines.

## About
This project was developed when I was an Insight AI Fellow, summer 2018. 
This package presently imports fluid dynamics simulation data and plots the best-defined features. 

## Contents
* YML: 
* config.config.yml has all of the user-set values for all of the functions is called by the main function.
* Python:
* RocketRL.python.main.py is the main function that calls the other scripts and data.
* RocketRL.python.func.data_processing.py imports the flow simulation data, plots the data, and creates a regression model.
* RocketRL.python.func.ray_funcs.py plots some of the outputs from the RL algorithm 
* RocketRL.python.envs.RocketRL.RocketEnv.py contains the custom env for this task.

## To run:

1. Clone or download this package
```bash
git clone https://github.com/ninalopatina/Rocket_RL.git
```

2. Install requirements from req.txt. (Note that this only works for OSX for now)

2b. If you're using conda, you can create a new environment with these requirements as below: 

```bash
$ conda create -n newenvironment --file req.txt
```
### Note: I just noticed that the below doesn't work correctly now; please check back after 7/10 for fixes.

3. Install Ray: 
- A few notes: Download the latest tag instead of the main branch b/c main branch doesn't build
```bash
git clone https://github.com/ray-project/ray.git
cd ray
git checkout tags/ray-0.4.0
```
- (Install instructions)[http://ray.readthedocs.io/en/latest/installation.html] 

4. Install RLlib:
- install snappy first if you don't already have it:
```bash
$ brew install snappy
```
-(Install instructions)[https://ray.readthedocs.io/en/latest/rllib.html]

5. In the root of this github repo, type :
```Bash
export PYTHONPATH=$PYTHONPATH:`pwd`/RocketRL/python
```

6. The contents of the env folder have to be copied to openAI gym envs folder for the rollout to work properly:
python3.6/site-packages/gym/envs/

7. In terminal, from the github repo folder Rocket_RL:
```Bash
source activate newenvironment
python RocketRL/python/main.py
```

8. To see your agent rendered in the rollout, run the below command in terminal, from the directory in which ray is installed. You will have to put the name of the folder in which the checkpoints are located in {folder_name}, for example, 'PPO_TwoTemp_0_2018-06-27_20-17-32lqfkwaol', and also indicate the checkpoint (that you have already trained past) #, i.e. 20, and number of steps, i.e. 50000

```Bash
python ray/python/ray/rllib/rollout.py ~/ray_results/RocketRL/{folder_name}/checkpoint-{checkpoint} --run PPO --env TwoTemp-v0 --steps {nsteps}
```

## To customize:
In the RocketEnv file, there are a few variables you can play with to get a feel for the model:

