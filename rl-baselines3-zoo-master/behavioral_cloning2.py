# https://imitation.readthedocs.io/en/latest/tutorials/1_train_bc.html
import gym
import panda_gym
#from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.ppo import PPO

from stable_baselines3.common.evaluation import evaluate_policy
#import assistive_gym
import datetime
import os
import numpy as np
import imageio
import json
import pickle

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from imitation.data.types import TrajectoryWithRew
from imitation.algorithms import bc
from stable_baselines3.common.logger import configure
import argparse
from PIL import Image

#from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
# https://stable-baselines.readthedocs.io/en/master/guide/examples.html   examples
# https://stable-baselines3.readthedocs.io/en/master/common/logger.html#logger  logs
'''
cd /home/codysoccerman/Documents/classes/Fall_22/Deep_Learning/Project/rl-baselines3-zoo-master
conda activate imitation
python3 behavioral_cloning2.py

cd /home/codysoccerman/Documents/classes/Fall_22/Deep_Learning/Project/rl-baselines3-zoo-master
tensorboard --logdir my_models/behavioral_cloning
http://localhost:6006/ 

'''

def flatten_observation(obs):
    a = np.array(obs['achieved_goal'])
    b = np.array(obs['desired_goal'])
    c = np.array(obs['observation'])
    obs_space_flat = np.concatenate((a,b,c),axis=0)
    return obs_space_flat

def save_video(image_array, directory, name):
    #print("saving video")
    video_writer = imageio.get_writer(directory + name, fps=20) #"videos/video_{}.mp4".format(episode), fps=20)
    for single_image in image_array:
        video_writer.append_data(np.asarray(single_image))
    video_writer.close()
    #print("video saved")

def save_trained_model(model, directory, name):
    print("saving model")
    
    model.save(directory+name+".zip")
    model.save(directory+name+".pt")
    #bc.save_policy(output_dir)
    #model.save_policy(output_dir+"model.pt")
    #model.save_model(output_dir) 
    #model.save_pretrained(model_path)
    
    #model.save_model(output_dir) #+"save/")

    pickle.dump(model, open(directory+name+"_pickle.pkl", 'wb'))
    
    #with open(output_dir + 'loss.txt', 'w') as convert_file:
    #    convert_file.write(str(loss_array))
    print("model saved")

def load_trained_model(model, checkpoint_path, name):
    print("loading model")
    #model = model.load(checkpoint_path + name + ".zip") #,env=env) # env=env
    #bc_trainer.policy = 
    model = pickle.load(open(checkpoint_path + name + "_pickle.pkl",'rb')) #works for training

    #model = PPO.load(models_dir+"/"+load_model_label+".zip",env=env)
    #model = MlpPolicy.load(model_path + environment,env=env)
    print("model loaded")
    return model

def load_expert_data(directory, name):

    demonstrations = pickle.load(open(directory + name + ".pkl",'rb'))
    action_space = gym.spaces.Box(
        np.array([-1.]*4, dtype=np.float16),
        np.array([1.]*4, dtype=np.float16)) 

    observation_space = gym.spaces.Box(
        np.array([-10]*25, dtype=np.float16),
        np.array([10]*25, dtype=np.float16))

    return demonstrations, action_space, observation_space

def load_expert_video_data(directory, name):

    demonstrations = pickle.load(open(directory + name + ".pkl",'rb'))
    action_space = gym.spaces.Box(
        np.array([-1.]*4, dtype=np.float16),
        np.array([1.]*4, dtype=np.float16)) 

    observation_space = gym.spaces.Box(
        np.array([0]*9216, dtype=np.float16),
        np.array([1]*9216, dtype=np.float16))

    return demonstrations, action_space, observation_space

def flatten_image(img, low_res, gray_image_array):
    img = img[:,:,:3]
    low_res_image = np.array(Image.fromarray(img).resize((low_res,low_res))) #96x96X3
    grayscaled_image = Image.fromarray(low_res_image).convert('L') #96x96x1
    gray_image_array.append(grayscaled_image) # built array
    single_image = np.asarray(grayscaled_image, dtype='float16') # convert to float 16
    obs_space_flat = np.reshape(single_image,(low_res*low_res))/255 #1x9216/255 1 frame [0-1] 
    return obs_space_flat, gray_image_array


'''
parser = argparse.ArgumentParser()
#parser.add_argument("--train_model", action="store_true", default=True, help="train_model")
#parser.add_argument("--save_model", action="store_true", default=True, help="save_model")
#parser.add_argument("--test_model", action="store_true", default=True, help="test_model")

parser.add_argument("--env", type=str, default="CartPole-v1", help="environment ID")
parser.add_argument("--use_video", action="store_true", default=True, help="use_video")
parser.add_argument("--num_epochs", type=int, default=10, help="num_epochs")


args = parser.parse_args()
print (args.train_model)
print (args.save_model)
'''

num_train_epochs = 1000
train_model = False
save_model = False
test_model = True
environment = "PandaPickAndPlace-v1"  #"PandaReach-v1" #"PandaPickAndPlace-v1"

use_video = False
model_path = "./my_models/behavioral_cloning/"
load_checkpoint = True
checkpoint_path = model_path +"11-23_06:28-200epochs/"

time = datetime.datetime.now().strftime('%m-%d_%H:%M')
output_dir = model_path+time+"-"+str(num_train_epochs)+"epochs/"



train_expert = False
enviroment_folder = "not assistive gym"
if train_expert == True:
    env = gym.make(environment) # "LunarLander-v2") # "CartPole-v1")
    print("env made")

    if load_model == True:
        time = load_model_label
    models_dir = "my_models/"+equation+"/"+environment #+"/"+time #+"/"
    logdir = "logs/"+equation+"/"+environment #+"/"+time #+"/"

    if train == True:
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        if not os.path.exists(logdir):
            os.makedirs(logdir)

    env.reset()

    model = PPO(policy=MlpPolicy, env=env, seed=0, batch_size=64, ent_coef=0.0, learning_rate=0.0003, n_epochs=100, n_steps=64, tensorboard_log=logdir)

    TIMESTEPS = 10
    if load_model == True:
        #model.load(models_dir+"/"+"50000.zip")
        model = PPO.load(models_dir+"/"+load_model_label+".zip",env=env) #, device="cpu")

    if train == True:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False) #, tb_log_name=f"PPO")  # Note: set to 100000 to train a proficient expert

    if save_model == True:
        model.save(models_dir+"/"+str(TIMESTEPS))

    '''
    # Save the agent
    print("Save the agent")
    model = bc_trainer.policy
    model_path = "my_models/trained_models/"
    model.save(model_path + environment)
    '''
    #print("test model")
    #reward, _ = evaluate_policy(model, env, 10) #10
    #print("model reward:", reward)

######################################################
# behavioral cloning 

test_rollout = False
if test_rollout == True:
    rng = np.random.default_rng()
    rollouts = rollout.rollout(model, DummyVecEnv([lambda: RolloutInfoWrapper(env)]), rollout.make_sample_until(min_timesteps=None, min_episodes=2), rng=rng)
    transitions = rollout.flatten_trajectories(rollouts)
    print(f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
    After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
    The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
    """)


# load expert demostrations
if use_video == True:
    demonstrations, action_space, obs_space = load_expert_video_data(model_path, environment) 
if use_video == False:
    demonstrations, action_space, obs_space = load_expert_data(model_path, environment) 

# define trainer settings
new_logger = configure(output_dir, ["csv", "tensorboard"]) #["stdout", "csv", "tensorboard"])
bc_trainer = bc.BC(observation_space=obs_space, action_space=action_space, demonstrations=demonstrations, device = "cuda", custom_logger = new_logger) #env.observation_space,  #env.action_space, transitions
model = bc_trainer.policy
# visualize bc
#print("bc_trainer.policy: ", bc_trainer.policy)

# Load the trained agent
if load_checkpoint == True:
    # https://imitation.readthedocs.io/en/latest/experts/loading-experts.html
    #model = load_trained_model(bc_trainer.policy, checkpoint_path, "model")
    bc_trainer = pickle.load(open(checkpoint_path + "bc_trainer_pickle.pkl",'rb'))
    #bc_trainer.policy = model

# Train BC
if train_model == True:
    bc_trainer.train(n_epochs=num_train_epochs, progress_bar = True) #, tensorboard_log=output_dir) #model.train(n_epochs=num_train_epochs)
model = bc_trainer.policy

# Save model
if save_model == True:
    save_trained_model(bc_trainer.policy, output_dir, "model")
    pickle.dump(bc_trainer, open(output_dir+"bc_trainer_pickle.pkl", 'wb'))

# Evaluate the agent
eval_agent = False
if eval_agent == True:  
    reward_after_training, _ = evaluate_policy(model, env, 10)
    print(f"Reward after training: {reward_after_training}")


# Test trained model
image_array = []
combined_array = []
gray_image_array = []
num_videos = 20
if test_model == True:
    env = gym.make(environment) # "LunarLander-v2") # "CartPole-v1")
    print("env made")
    os.makedirs(output_dir + "videos/")

    for episode in range(num_videos):

        episode_reward = 0
        obs = env.reset()
        while True:
            img = env.render("rgb_array")

            if use_video == False:
                obs_space_flat = flatten_observation(obs)
            if use_video == True:
                #img = env.render("rgb_array") #rgba env.render("human") ##
                obs_space_flat, gray_image_array = flatten_image(img, 96, gray_image_array)

            action, _states = model.predict(obs_space_flat)

            #print("action: ", action)
            #action = [0.01, 0.0, 0.0, 0.0]
            #img = env.render("rgb_array")  #env.render()
            image_array.append(img)
            combined_array.append(img)

            obs, rewards, dones, info = env.step(action)
            episode_reward = episode_reward + rewards
            if dones == True:
                save_video(image_array, output_dir, "videos/video_{}.mp4".format(episode))
                image_array = []
                env.reset()
                print("episode: ",episode, " reward: ", episode_reward)
                break

    env.close()    

    print("save combined video")
    save_video(combined_array, output_dir, "videos/videos_combined.mp4")

    #print("save combined gif")
    #imageio.mimsave(output_dir + 'videos/gifs_combined.gif', combined_array)
    
