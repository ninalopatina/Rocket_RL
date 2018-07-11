# Rocket_RL
### Reinforcement learning agent to automate rocket engine tuning

## About
This project was developed when I was an Insight AI Fellow, summer 2018. 
This package imports fluid dynamics simulation data, creates a regression model mapping inputs --> outputs, then trains an RL agent to derive inputs to satisfy output conditions. (Demo slides)[http://goo.gl/hksviY]

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
$ git clone https://github.com/ninalopatina/Rocket_RL.git
```

2. Install requirements from req.txt. (Note that this only works for OSX for now). Instructions below are for installing in a new environment; you can skip the first 2 lines below if you wish to install this in an existing environment .

```bash
$ conda create -n newenv python=3
$ source activate newenv
$ pip install -r Rocket_RL/req.txt
```
### Note: I just noticed that the below doesn't work correctly now; please check back after 7/10 for fixes.

###Note: I just noticed that the below doesn't work correctly now; please check back after 7/10 for fixes.

3. Install Ray: 
- Note: Instructions below are to download the latest tag instead of the main branch b/c main branch doesn't build. This install takes a while. 
```bash
$ git clone https://github.com/ray-project/ray.git
$ cd ray
$ git checkout tags/ray-0.4.0
$ cd python
```

The below is from their instructions:

```bash
brew update
brew install cmake pkg-config automake autoconf libtool openssl bison wget
pip install cython
```

If you are using Anaconda:
```bash
conda install libgcc
```


$ pip install -e . --verbose
cd ..
python test/runtest.py ****note to self: stuck here
```
- (Install instructions modified from)[http://ray.readthedocs.io/en/latest/installation.html] 

4. Install RLlib:
- install snappy first if you don't already have it:
```bash
$ brew install snappy
$ pip install ray[rllib]
```
-(Install instructions adapted from)[https://ray.readthedocs.io/en/latest/rllib.html]

5. Go back to the root of this github repo:
```bash
$ cd ../..

6. In the root of this github repo, type :
```Bash
$ export PYTHONPATH=$PYTHONPATH:`pwd`/RocketRL/python
```

7. The contents of the Rocket_RL/RocketRL/python/envs folder have to be copied to openAI gym envs folder for the rollout to work properly: python3.6/site-packages/gym/envs/

8. In terminal, from the github repo folder Rocket_RL:
```Bash
$ python RocketRL/python/main.py
```

9. To see your agent rendered in the rollout, run the below command in terminal, from the directory in which ray is installed. You will have to put the name of the folder in which the checkpoints are located in {folder_name}, for example, 'PPO_TwoTemp_0_2018-06-27_20-17-32lqfkwaol', and also indicate the checkpoint (that you have already trained past) #, i.e. 20, and number of steps, i.e. 50000

```Bash
python ray/python/ray/rllib/rollout.py ~/ray_results/RocketRL/{folder_name}/checkpoint-{checkpoint} --run PPO --env TwoTemp-v0 --steps {nsteps}
```

## To customize:
In the RocketEnv file, there are a few variables you can play with to get a feel for the model:

