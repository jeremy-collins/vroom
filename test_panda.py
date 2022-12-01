# https://imitation.readthedocs.io/en/latest/tutorials/1_train_bc.html
import gym
import panda_gym
import torch
#from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.ppo import PPO

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
# from stable_baselines3.common.policies import obs_to_tensor
#import assistive_gym
import datetime
import os
import numpy as np
import imageio
import json
import pickle
import numpy as np
import copy

from bc_mlp import BC_custom

'''
cd /home/codysoccerman/Documents/classes/Fall_22/Deep_Learning/Project/rl-baselines3-zoo-master
conda activate assistive
python3 behavioral_cloning2.py

tensorboard --logdir logs/ppo/BedBathingStretch-v1
http://localhost:6006/

'''
enviroment_folder = "aoa" #"assistive gym"
environment = "PandaPickAndPlace-v1" #"CartPole-v1" #"PandaPickAndPlace-v1" #"CartPole-v1" #"BedBathingStretch-v1"# DrinkingStretch-v1 # #FetchReach-v1"   #LunarLander-v2" # FetchSlide-V1
equation = "ppo"
load_model = False
load_model_label = "10000"
train = False
record_video = False
save_model = True

env = gym.make(environment, render=True) # "LunarLander-v2") # "CartPole-v1")
print("env made")

time = datetime.datetime.now().strftime('%m-%d_%H-%M')
if load_model == True:
    time = load_model_label
models_dir = "my_models/"+equation+"/"+environment #+"/"+time #+"/"
logdir = "logs/"+equation+"/"+environment #+"/"+time #+"/"


model = BC_custom(input_size=25, output_size=4, net_arch=[32,32], extractor='lstm')

# Load the trained agent
print("Load the agent")
model.load_state_dict(torch.load('checkpoints/model_pandlstm_2.pt'))
# model = MlpPolicy.load('./results/11-23_06_28-200epochs/model.pt')
# model = PPO.load('/home/alanhesu/Documents/github/vroom/rl-baselines3-zoo-master/my_models/backup/PandaPickAndPlace-v1.pkl', env=env)


# Enjoy trained agent
episode_reward = 0
X_seq = []
X_seq.append(torch.zeros(25))
X_seq.append(torch.zeros(25))
obs = env.reset()

reward_list = []
for i in range(1000):
    X = copy.deepcopy(obs)
    X = np.concatenate((X['achieved_goal'], X['desired_goal'], X['observation']))
    X = torch.from_numpy(X)
    X = X.float()

    # X_input = X # for flatten

    X_seq.append(X) # for lstm
    X_seq = X_seq[1:]
    X_input = torch.stack(X_seq, dim=0)

    X_input = X_input[None,:]

    action, _, _ = model(X_input)
    action = torch.squeeze(action, dim=0)
    action = action.detach().numpy()

    # action, _ = model.predict(obs)
    env.render()
    obs, rewards, dones, info = env.step(action)
    episode_reward = episode_reward + rewards
    if dones:
        print("episode_reward: ", episode_reward)
        reward_list.append(episode_reward)
        episode_reward = 0
        print("reset")
        obs = env.reset()

        #time.sleep(1/30)

print('average reward: {}'.format(np.mean(reward_list)))
print('num trials: {}'.format(len(reward_list)))
env.close()
