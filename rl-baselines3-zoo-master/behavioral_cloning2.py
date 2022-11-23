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

'''
cd /home/codysoccerman/Documents/classes/Fall_22/Deep_Learning/Project/rl-baselines3-zoo-master
conda activate assistive
python3 behavioral_cloning2.py

tensorboard --logdir logs/ppo/BedBathingStretch-v1
http://localhost:6006/ 

'''

#model =  <class 'stable_baselines3.ppo.ppo.PPO'> .load( rl-trained-agents/ppo/LunarLander-v2_1/LunarLander-v2.zip , env= <stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv object at 0x7f54056c84f0> , custom_objects= {'learning_rate': 0.0, 'lr_schedule': <function enjoy.<locals>.<lambda> at 0x7f54056d00d0>, 'clip_range': <function enjoy.<locals>.<lambda> at 0x7f54056d0160>} , device= auto , ,**kwargs)

#model_path = 'rl-trained-agents/ppo/LunarLander-v2_1/LunarLander-v2.zip'
#model = PPO.load(model_path) #, env=env, custom_objects=custom_objects, device=args.device, **kwargs)

#reward_after_training, _ = evaluate_policy(model, env, 10)
#print(f"Reward after training: {reward_after_training}")

####################################################
enviroment_folder = "aoa" #"assistive gym"
environment = "PandaPickAndPlace-v1" #"CartPole-v1" #"PandaPickAndPlace-v1" #"CartPole-v1" #"BedBathingStretch-v1"# DrinkingStretch-v1 # #FetchReach-v1"   #LunarLander-v2" # FetchSlide-V1
equation = "ppo"
load_model = False
load_model_label = "10000"
train = False
record_video = False
save_model = True

env = gym.make(environment) # "LunarLander-v2") # "CartPole-v1")
print("env made")

time = datetime.datetime.now().strftime('%m-%d_%H-%M')
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

model = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=100,
    n_steps=64,
    tensorboard_log=logdir,
)

TIMESTEPS = 10
if load_model == True:
    #model.load(models_dir+"/"+"50000.zip")
    print("loading from: ", models_dir+"/"+load_model_label+".zip")
    model = PPO.load(models_dir+"/"+load_model_label+".zip",env=env) #, device="cpu")
    print("model loaded")

if train == True:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False) #, tb_log_name=f"PPO")  # Note: set to 100000 to train a proficient expert
    print("model trained")

if save_model == True:
    model.save(models_dir+"/"+str(TIMESTEPS))
    print("model saved")

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

#from numpngw import write_apng
#from IPython.display import Image
if record_video == True:
    print("making video")
    if enviroment_folder == "assistive gym":
        env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4)

    # Reset the environment
    observation = env.reset()
    frames = []
    done = False
    episode_reward = 0
    for i in range(1000):
        if enviroment_folder != "assistive gym":
            img = env.render(mode='rgb_array')
        
        action, _states = model.predict(observation)

        observation, reward, done, info = env.step(action)
        episode_reward = episode_reward + reward
        
        if enviroment_folder == "assistive gym":
            img, depth = env.get_camera_image_depth()
        frames.append(img)
        #write_apng('output.png', frames, delay=100)
        #Image(filename='output.png')
    
        if done:
            print("episode_reward: ", episode_reward)
            episode_reward = 0
            print("reset")
            observation = env.reset()
            #time.sleep(1/30)


    if enviroment_folder == "assistive gym":
        env.disconnect()

    video_writer = imageio.get_writer(models_dir+"/video_{}.mp4".format(TIMESTEPS), fps=20)
    for frame in frames:
        video_writer.append_data(np.asarray(frame))
    video_writer.close()
    print("video saved")

######################################################

# behavioral cloning below

######################################################

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from imitation.data.types import TrajectoryWithRew
#from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
'''
expert = model
rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=2),
    rng=rng,
)

#print("rollouts: ", rollouts)
#print("rollouts[acts]: ", rollouts["acts"])
#print("rollouts[0]: ", rollouts)
#print("rollouts[0]['acts']: ", rollouts[0]["acts"])
transitions = rollout.flatten_trajectories(rollouts)

###########################################################
print(type(rollouts))
print(type(transitions))
#print(transitions)
#with open('decision_transformer_gym_replay/transitions.txt', 'w') as convert_file:
#    convert_file.write(json.dumps(transitions))

print(
    f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
"""
)

'''

obj_test = TrajectoryWithRew(obs=np.array([[-9.58364177e-03, -1.24337887e-02,  2.88221259e-02,-3.22786830e-02],
                                  [-9.83231794e-03,  1.82263240e-01,  2.81765517e-02,-3.15730393e-01]]), 
                                  acts=np.array([1]),
                                  infos=None, terminal=True, rews=np.array([1.]) )
demonstrations = pickle.load(open("my_models/trained_models/PandaPickAndPlace-v1.pkl",'rb'))
#print(demonstrations)
#print("obj_test: ", obj_test)
print("env.observation_space: ", env.observation_space)
print("env.action_space: ", env.action_space)

action_space = gym.spaces.Box(
    np.array([-1.,-1.,-1.,-1.], dtype=np.float32),   #, -1, -1, -1, -1], dtype=np.float32), 
    np.array([1.,1.,1.,1.], dtype=np.float32)) #  , 1, 1, 1, 1], dtype=np.float32))
observation_space = gym.spaces.Box(
    np.array([-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10], dtype=np.float32),   #, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
    np.array([10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10], dtype=np.float32))


'''
action_space = gym.spaces.Discrete(2) #  , 1, 1, 1, 1], dtype=np.float32))
observation_space = gym.spaces.Box(
        np.array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38], dtype=np.float32),   #, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
        np.array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38], dtype=np.float32))

print("action_space: ", action_space)
print("observation_space: ", observation_space)
'''
###########################################################

from imitation.algorithms import bc

bc_trainer = bc.BC(
    observation_space=observation_space, #env.observation_space,
    action_space=action_space, #env.action_space,
    demonstrations=demonstrations,
    device = "cuda",
)
#rng=rng,
#reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
#print(f"Reward before training: {reward_before_training}")

# train bc
bc_trainer.train(n_epochs=1000)
reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward after training: {reward_after_training}")




# Save the agent
print("Save the agent")
model = bc_trainer.policy
model_path = "my_models/trained_models/"
model.save(model_path + environment)

del model  # delete trained model to demonstrate loading

# Load the trained agent
print("Load the agent")
model = MlpPolicy.load(model_path + environment)

# Evaluate the agent
print("Evaluate the Loaded agent")
reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward after training: {reward_after_training}")



'''
# Load the trained agent
model = MlpPolicy.load(model_path + environment,,env=env)

# Evaluate the agent
reward_after_training, _ = evaluate_policy(model, env, 10)
print(f"Reward after training: {reward_after_training}")
'''


# Enjoy trained agent
episode_reward = 0
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    env.render()
    obs, rewards, dones, info = env.step(action)
    episode_reward = episode_reward + rewards
    if dones:
        print("episode_reward: ", episode_reward)
        episode_reward = 0
        print("reset")
        obs = env.reset()
        #time.sleep(1/30)

env.close()    





#########################################################
# https://stable-baselines.readthedocs.io/en/master/guide/examples.html
'''
import pickle

model_name = "behavioral_cloning"
model_path = "my_models/trained_models/" + model_name + ".pickle"

print("save model")
pickle.dump(bc_trainer.policy, open(model_path, 'wb'))

print("load saved model")
model = pickle.load(open(model_path,'rb'))
reward_after_training, _ = evaluate_policy(model, env, 100)

print(f"Reward after training: {reward_after_training}")
'''
###########################################################
