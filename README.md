# Rocket_RL
### Reinforcement learning agent to automate rocket engine tuning

## About
This project was developed when I was an Insight AI Fellow, summer 2018. 
This package imports fluid dynamics simulation data, creates a regression model mapping inputs --> outputs, then trains an RL agent to derive inputs to satisfy output conditions. (Demo slides)[http://goo.gl/hksviY]

## Contents
* YML: config.config.yml has all of the user-set values for all of the functions. It is called by the main function.

* Data: data.rocket_data.csv contains cached flow simulation data collected from Fluent. These data were used to create a polynomial linear regression model of the fluid dynamics simulation that would be run to tune a rocket engine. This model is the "interpreter" that the OpenAI gym environment calls. 

* Python:
* python.main.py is the main function that calls the other functions and data.
* python.func.data_processing.py imports the flow simulation data, plots the data, and creates a regression model.
* python.func.ray_funcs.py plots some of the outputs from the RL algorithm.
* python.envs.RocketRL.RocketEnv.py contains the custom env for this task.
* python.envs.__init.py__ registers the custom environment. It sets the max_episode_steps.

* Pickles: results.pickles contains the pickles of regression powers, coefficients, and intercept from the model I trained.

* video_labels contains the .png files that the rollout rendering displays.
 
* req.txt contains the requirements to run the Rocket_RL package, excluding Ray & RLlib, whose specific installation instructions are below.

## To run:

1. Clone or download this package
```bash
$ git clone https://github.com/ninalopatina/Rocket_RL.git
```

2. Install requirements (Note that this only works for OSX for now). 

A) Instructions below are for installing in a new conda environment. *This is recommended

```bash
$ conda create -n newenv --file req-conda.txt python=3
$ source activate newenv
```

B) Or you can just pip install all the requirements: 

```bash
$ pip install -r Rocket_RL/req.txt
```
### Note: I just noticed that the below doesn't work correctly now; please check back after 7/10 for fixes.

3. Install Ray: 
- Note: Instructions below are adapted from their (instructions)[http://ray.readthedocs.io/en/latest/installation.html], with some additions to download the latest tag instead of the main branch b/c main branch doesn't build. This install takes a while. 

```bash
$ brew update
$ brew install cmake pkg-config automake autoconf libtool openssl bison wget
$ pip install cython
```

If you are using Anaconda:
```bash
$ conda install libgcc
```

Modification to install the latest tag (as of 7/10/18)
```bash
$ brew install snappy
$ git clone https://github.com/ray-project/ray.git
$ cd ray
$ git checkout tags/ray-0.4.0
```

Continuing their instructions:

```bash
$ cd python
$ pip install -e . --verbose
$ cd ..
$ python test/runtest.py 
```

4. Install RLlib:

```bash
$ cd python
$ pip install ray[rllib]
```

5. Install gym. For some reason it didn't install from req.txt when I tested this out. 
```bash
$ pip install gym
```

6. Go back to the root of the directory containing the Rocket_RL repo and Ray & export the Rocket_RL python path:

```bash
$ cd ../..
$ export PYTHONPATH=$PYTHONPATH:`pwd`Rocket_RL/python
```

7. The contents of the Rocket_RL/python/envs folder have to be copied to openAI gym envs folder for the rollout to work properly: python3.6/site-packages/gym/envs/

8. In terminal, from the directory containing Rocket_RL, train the model with:

```Bash
$ python Rocket_RL/python/main.py
```

* Note: --save_reg is set to False by default, so you can use the regression parameters I used in my trained model. If you would like to change these, make sure you don't save new regression variables between training (above) and rollout (below).

9. To rollout your trained agent, run the below command in terminal. You will have to put the name of the folder in which the checkpoints are located in {folder_name}, for example, 'PPO_All_Var_0_2018-06-27_20-17-32lqfkwaol', and also indicate the {checkpoint} (that you have already trained past) #, i.e. 20, and number of steps, {nsteps}, i.e. 50000

```Bash
python ray/python/ray/rllib/rollout.py ~/ray_results/RocketRL/{folder_name}/checkpoint-{checkpoint} --run PPO --env AllVar-v0 --steps {nsteps}
```

The above should show a rendering of the trained agent in action. 

## To customize the model:
There are a few variables you can change to get a feel for the RL model:

- In config.config.yml:

* For the regression model, you can try different degrees for the polynomial features that the model considers. It will save degree_max. I haven't gotten degree_max = 4 to work, so let me know if you do!

- In python.envs.RocketRL.RocketEnv.py:


- In python.envs.__init.py__
max_episode_steps - this is the maximum number of steps that the agent takes before failing at an episode. One of the limitations to PPO is that it is easy for the agent to fail to reach the target. If the max steps are too high, the agent will meander aimlessly, and learn a weird superstition that it will attempt to follow later. If the number is too low, the agent doesn't have a chance to learn how to reach the target. I have erred on both sides of this. Setting this to 1000 works well with the other parameters I set. 


