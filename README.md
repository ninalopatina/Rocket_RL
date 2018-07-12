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

1. Clone or download this package into your home directory:

```bash
$ git clone https://github.com/ninalopatina/Rocket_RL.git
```

2. Install requirements (Note that this only works for OSX for now). 

A) In a new conda environment. ****This is highly recommended

```bash
$ conda create -n RocketEnv --file Rocket_RL/req-conda.txt python=3.6
$ source activate RocketEnv
$ pip install cython
$ pip install gym
$ pip install pyyaml
```

B) Or pip install. ** I'm not sure if this works. I highly recommend option A above.

```bash
$ pip install -r Rocket_RL/req.txt

```

3. Install Ray: 
- Note: Instructions below are from (Ray docs)[http://ray.readthedocs.io/en/latest/installation.html]. This install takes a while. 

```bash
$ brew update
$ brew install cmake pkg-config automake autoconf libtool openssl bison wget snappy
```

If you are using Anaconda:

```bash
$ conda install libgcc
$ conda install pytorch torchvision -c pytorch
```

Continuing:
```bash
$ git clone https://github.com/ray-project/ray.git
$ cd ray/python
$ pip install -e . --verbose
$ cd ..
$ python test/runtest.py 
```

4. Install RLlib:

```bash
$ cd python
$ pip install ray[rllib]
```

5. Go back to the root of the home directory and then export the Rocket_RL python path:

```bash
$ cd ../..
$ export PYTHONPATH=$PYTHONPATH:`pwd`/Rocket_RL/python
```

6. The contents of the Rocket_RL/python/envs folder have to be manually copied to the openAI gym envs folder for the rollout to work properly: so please copy the contents to python3.6/site-packages/gym/envs/ in the environment you're running this in

7. In terminal, from the home directory, train the model with:

```Bash
$ python Rocket_RL/python/main.py
```

* Note: --save_reg is set to False by default, so you can use the regression parameters I used in my trained model. If you would like to change these, make sure you don't save new regression variables between training (above) and rollout (below).

8. To rollout your trained agent, run the below command in terminal. You will have to put the name of the folder in which the checkpoints are located in {folder_name}, for example, 'PPO_All_Var_0_2018-06-27_20-17-32lqfkwaol', and also indicate the {checkpoint} (that you have already trained past) #, i.e. 20, and number of steps, {nsteps}, i.e. 50000

```Bash
python ray/python/ray/rllib/rollout.py ~/ray_results/RocketRL/{folder_name}/checkpoint-{checkpoint} --run PPO --env AllVar-v0 --steps {nsteps}
```

The above should show a rendering of the trained agent in action. 

## To customize the model:
There are a few variables you can change to get a feel for the RL model:

### In config.config.yml:

#### For the regression model:
* reg_model: sklearn's linear regression, or Lasso CV. I prefer lasso because the L1 regularization penalizes the features, which generates an easier model for the agent to solve. The accuracy is indistinguishable from linreg. 
* degree_max: You can try different degrees for the polynomial features that the model considers. It will save degree_max. I haven't gotten degree_max = 4 to work, so let me know if you do!

#### For the RL model:

##### Within the environment:
* scale_var: Scaling factor for all the inputs and outputs. Note that all of the below are multiplied by this variable, so they are expressed as a decimal. I.E., action_range is the percentage of the range that the actions can span. The interpreter was run on normalized data, so, this increases the scale. 

* reward: This is the reward the agent receives upon reaching the goal. If it's not large enough, the reward increase of reaching the goal will be washed out by the negative rewards the agent accumulated in the preceding steps.

* thresh (1-3): Scales the MSE when calculating the negative reward. Since the MSE is small, this scales it up so the agent can better distinguish between states

* action range: This is a continuous action space. The range is the amount the agent can go up or down in the action space. If it's too small, the agent will have to take too many steps to reach the goal, even if it knows where to go. The PPO agent will sample a smaller action space than this full range when it starts learning. 

* noise: Noise is really an add-on to the regression model: it adds some random noise to the deterministic regression output to more closely mimic the simulation space. The range is 0-1. When prototyping new features, it helps to set 
this to 0 to identify how the variable changes you made affect the outcome, and then adding it back in to see if it is robust to noise. 

*min_max_buffer: This removes the top & bottom fraction of the input space. The range is 0-1. This is also a feature for prototyping, to simplify the problem. 

##### Experiment conditions:
* Set the total time, total timesteps, or reward mean

##### PPO agent variables: 
* gamma: discount factor of the Markov Decision Process
* horizon: steps before rollout is cut
* num_sgd_iter: # iterations in each outer loop. Stochastic gradient descent
* timesteps_per_batch: 
* min_steps_per_task: 

### In python.envs.__init.py__
* max_episode_steps - this is the maximum number of steps that the agent takes before failing at an episode. One of the limitations to PPO is that it is easy for the agent to fail to reach the target. If the max steps are too high, the agent will meander aimlessly, and learn a weird superstition that it will attempt to follow later. If the number is too low, the agent doesn't have a chance to learn how to reach the target. I have erred on both sides of this. Setting this to 1000 works well with the other parameters I set. 


