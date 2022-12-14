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
import argparse
import cv2

from bc_mlp import BC_custom
from ili_transformer.transformer_timeseries import TimeSeriesTransformer

# python test_panda.py --modelfile checkpoints/model_pandmagic_lr1e-4_.1epochratio.pt --modeltype magicalcnn

'''
cd /home/codysoccerman/Documents/classes/Fall_22/Deep_Learning/Project/rl-baselines3-zoo-master
conda activate assistive
python3 behavioral_cloning2.py

tensorboard --logdir logs/ppo/BedBathingStretch-v1
http://localhost:6006/

'''
def test_env(modelfile, modeltype, frame_size=(96,96), frames_per_clip=1):
    environment = "PandaPickAndPlace-v1" #"CartPole-v1" #"PandaPickAndPlace-v1" #"CartPole-v1" #"BedBathingStretch-v1"# DrinkingStretch-v1 # #FetchReach-v1"   #LunarLander-v2" # FetchSlide-V1

    env = gym.make(environment, render=True) # "LunarLander-v2") # "CartPole-v1")
    print("env made")

    if (modeltype == 'mlp'):
        model = BC_custom(input_size=25, output_size=4, net_arch=[32,32], extractor='flatten')
    elif (modeltype == 'cnn'):
        model = BC_custom(input_size=2048, output_size=4, net_arch=[32,32], extractor='cnn2')
    elif (modeltype == 'magicalcnn'):
        model = BC_custom(input_size=128, output_size=4, net_arch=[32,32], extractor='magicalcnn')
    elif (modeltype == 'lstm'):
        model = BC_custom(input_size=25, output_size=4, net_arch=[32,32], extractor='lstm')
    elif (args.modeltype == 'transformer'):
        model = BC_custom(input_size=25, output_size=4, net_arch=[32,32], extractor='transformer', num_frames=frames_per_clip)
    elif (args.modeltype == 'magicalcnnlstm'):
        model = BC_custom(input_size=128, output_size=4, net_arch=[32,32], extractor='magicalcnnlstm', freeze_cnn=False)
    elif (args.modeltype == 'magicalcnntransformer'):
        model = BC_custom(input_size=128, output_size=4, net_arch=[32,32], extractor='magicalcnntransformer', freeze_cnn=False, num_frames=frames_per_clip)
    else:
        print('modeltype {} not supported'.format(modeltype))

    # Load the trained agent
    print("Load the agent")
    model.load_state_dict(torch.load(modelfile))
    # model = MlpPolicy.load('./results/11-23_06_28-200epochs/model.pt')
    # model = PPO.load('/home/alanhesu/Documents/github/vroom/rl-baselines3-zoo-master/my_models/backup/PandaPickAndPlace-v1.pkl', env=env)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.eval()
    # Enjoy trained agent
    episode_reward = 0
    # for i in range(0, frames_per_clip):
    #     X_seq.append(torch.zeros(25))
    obs = env.reset()

    reward_list = []
    X_seq = reset_seq(frames_per_clip, modeltype, frame_size)
    for i in range(10000):
        X = copy.deepcopy(obs)
        X = np.concatenate((X['achieved_goal'], X['desired_goal'], X['observation']))
        X = torch.from_numpy(X)
        X = X.float()

        if (modeltype == 'mlp'):
            X_input = X # for flatten
        elif (modeltype == 'lstm' or modeltype == 'transformer'):
            X_seq.append(X) # for lstm
            if (len(X_seq) > frames_per_clip):
                X_seq.pop(0)
            X_input = torch.stack(X_seq, dim=0)
        elif (modeltype == 'cnn' or modeltype == 'magicalcnn' or modeltype == 'magicalcnnlstm' or modeltype == 'magicalcnntransformer'):
            X_input = env.render('rgb_array')[:,:,:3]
            X_input = cv2.resize(X_input, frame_size)
            # let's swap the channels
            ch1 = X_input[:,:,0].copy()
            X_input[:,:,0] = X_input[:,:,2]
            X_input[:,:,2] = ch1
            X_input = torch.from_numpy(X_input)
            X_input = X_input.permute(2, 0, 1)
            X_input = X_input.float() / 255.0
            if (modeltype == 'magicalcnnlstm' or modeltype == 'magicalcnntransformer'):
                X_seq.append(X_input)
                if (len(X_seq) > frames_per_clip):
                    X_seq.pop(0)
                X_input = torch.stack(X_seq, dim=0)

        X_input = X_input[None,:]
        X_input = X_input.to(device)

        action, _, _ = model(X_input)
        action = torch.squeeze(action, dim=0)
        action = action.detach().cpu().numpy()

        # action, _ = model.predict(obs)
        # env.render()
        obs, rewards, dones, info = env.step(action)
        episode_reward = episode_reward + rewards
        if dones:
            # print("episode_reward: ", episode_reward)
            reward_list.append(episode_reward)
            episode_reward = 0
            # print("reset")
            obs = env.reset()
            X_seq = reset_seq(frames_per_clip, modeltype, frame_size)

            #time.sleep(1/30)

    print('average reward: {}'.format(np.mean(reward_list)))
    print('num trials: {}'.format(len(reward_list)))
    print('number successes: {}'.format(len(reward_list) - np.sum(np.array(reward_list) == -50.0)))
    env.close()

def reset_seq(num_frames, modeltype, frame_size):
    X_seq = []
    for i in range(0, num_frames):
        if (modeltype == 'lstm' or modeltype == 'transformer'):
            X_seq.append(torch.zeros(25).float())
        elif (modeltype == 'magicalcnnlstm' or modeltype == 'magicalcnntransformer'):
            X_seq.append(torch.zeros((3, frame_size[0], frame_size[1])).float())

    return X_seq

frame_size = (96, 96)
frames_per_clip = 5

parser = argparse.ArgumentParser()
parser.add_argument('--modelfile', type=str, required=True)
parser.add_argument('--modeltype', type=str, required=True)
args = parser.parse_args()

test_env(args.modelfile, args.modeltype, frame_size=frame_size, frames_per_clip=frames_per_clip)