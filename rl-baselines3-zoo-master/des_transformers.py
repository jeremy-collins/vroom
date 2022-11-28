import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments, AutoModel
import gym
import panda_gym
import pickle
import datetime
from PIL import Image
import cv2
import skvideo.io
import imageio
#import cardinality
import shutil
import json
import mujoco_py
from colabgymrender.recorder import Recorder
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

'''
cd /home/codysoccerman/Documents/classes/Fall_22/Deep_Learning/Project/rl-baselines3-zoo-master
conda activate imitation
python3 des_transformers.py


cd /home/codysoccerman/Documents/classes/Fall_22/Deep_Learning/Project/rl-baselines3-zoo-master
tensorboard --logdir decision_transformer_gym_replay/trained_models/
'''


def flatten_observation(obs):
    a = np.array(obs['achieved_goal'])
    b = np.array(obs['desired_goal'])
    c = np.array(obs['observation'])
    obs_space_flat = np.concatenate((a,b,c),axis=0)
    return obs_space_flat

def save_video(image_array, directory, name):
    print("saving video")
    video_writer = imageio.get_writer(directory + name, fps=20) #"videos/video_{}.mp4".format(episode), fps=20)
    for single_image in image_array:
        video_writer.append_data(np.asarray(single_image))
    video_writer.close()
    print("video saved")

def flatten_image(img, low_res, gray_image_array):
    img = img[:,:,:3]
    low_res_image = np.array(Image.fromarray(img).resize((low_res,low_res))) #96x96X3
    grayscaled_image = Image.fromarray(low_res_image).convert('L') #96x96x1
    gray_image_array.append(grayscaled_image) # built array
    single_image = np.asarray(grayscaled_image, dtype='float16') # convert to float 16
    obs_space_flat = np.reshape(single_image,(low_res*low_res))/255 #1x9216/255 1 frame [0-1] 
    return obs_space_flat, gray_image_array


#load_trained_model = False
model_path = "./decision_transformer_gym_replay/trained_models/"
num_train_epochs = 2
use_video = True
train_model = True
save_model = True
delete_old_data = False
enviroment = "PandaPickAndPlace-v1"  #"PandaReach-v1" #"PandaPickAndPlace-v1"
load_checkpoint = False
checkpoint_path = model_path +"nov22_100solves_500epochs/"
checkpoint = "checkpoint-1500"
test_model = False

if delete_old_data == True:
    #home/codysoccerman/.cache/huggingface/datasets/decision_transformer_gym_replay/PandaPickAndPlace-v1/
    if os.path.exists("/home/codysoccerman/.cache/huggingface/datasets/decision_transformer_gym_replay/" + enviroment + "/1.1.0"): #/datasets/decision_transformer_gym_replay/" + env_name):
        print("exists")
        shutil.rmtree("/home/codysoccerman/.cache/huggingface/datasets/decision_transformer_gym_replay/" + enviroment + "/1.1.0")      #PandaPickAndPlace-v1/1.1.0")
        print("old files deleted")
    else:
        print("no files to delete")


print("start")
# import torch
print("is gpu available: ", torch.cuda.is_available())

os.environ["WANDB_DISABLED"] = "true" # we diable weights and biases logging for this tutorial
dataset = load_dataset("decision_transformer_gym_replay", enviroment) #, keep_in_memory=False)  #keep_in_memory=True, # "halfcheetah-expert-v2") #"PandaPickAndPlace-v1")
print(type(dataset))
print(dataset)

# act_dim:  4 type:  <class 'list'>
# state_dim:  9216 type:  <class 'list'>
# obs:  50 type:  <class 'list'>

# Step 4: Defining a custom DataCollator for the transformers Trainer class
#@dataclass
class DecisionTransformerGymDataCollator:
    return_tensors: str = "pt"
    max_len: int = 25 #subsets of the episode we use for training
    if use_video == True:
        state_dim: int = 9216  # size of state space
    else:
        state_dim: int = 25 #9216  # size of state space
    act_dim: int = 4  # size of action space
    max_ep_len: int = 50 # max episode length in the dataset
    scale: float = 1 #1000.0  # normalization of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0 # to store the number of trajectories in the dataset

    def __init__(self, dataset) -> None:
        self.act_dim = len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["observations"][0])
        self.dataset = dataset
        # calculate dataset stats for normalization of states
        states = []
        traj_lens = []
        print("for loop") # only 1 pass, slow remove if possible
        for obs in dataset["observations"]:
            states.extend(obs)
            traj_lens.append(len(obs))
        self.n_traj = len(traj_lens)
        states = np.vstack(states)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        
        traj_lens = np.array(traj_lens)
        self.p_sample = traj_lens / sum(traj_lens)

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        #print("GymDataCollator start __call__")
        batch_size = len(features)
        # this is a bit of a hack to be able to sample of a non-uniform distribution
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )
        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        #print("GymDataCollator start: for loop")
        #with open('decision_transformer_gym_replay/self.dataset.txt', 'w') as convert_file:
        #    convert_file.write(json.dumps(self.dataset[0]))

        for ind in batch_inds:
            # for feature in features:

            feature = self.dataset[int(ind)]
            si = random.randint(0, len(feature["rewards"]) - 1)

            #with open('decision_transformer_gym_replay/s1.txt', 'w') as convert_file:
            #    convert_file.write(str(feature["states"]))
            # get sequences from dataset
            #print("for loop s start")
            s.append(np.array(feature["observations"][si : si + self.max_len]).reshape(1, -1, self.state_dim))
            #print("for loop s end")
            #with open('decision_transformer_gym_replay/s1b.txt', 'w') as convert_file:
            #    convert_file.write(str(s))

            a.append(np.array(feature["actions"][si : si + self.max_len]).reshape(1, -1, self.act_dim))
            r.append(np.array(feature["rewards"][si : si + self.max_len]).reshape(1, -1, 1))

            d.append(np.array(feature["dones"][si : si + self.max_len]).reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature["rewards"][si:]), gamma=1.0)[
                    : s[-1].shape[1]   # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                #print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            #print("tlen: ", tlen)
            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))
        #print("GymDataCollator end: for loop")
        #print("inside GymDataCollator end: for ind in batch_inds")
        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()
        #print("DecisionTransformerGymDataCollator return")
        #print("GymDataCollator end __call__")
        
        #print("actions shape: ", a.cpu().detach().numpy().shape)
        
        #with open('decision_transformer_gym_replay/s2.txt', 'w') as convert_file:
        #    convert_file.write(np.array2string(s.cpu().detach().numpy()))
        '''
        with open('decision_transformer_gym_replay/notes/a.txt', 'w') as convert_file:
            convert_file.write(np.array2string(a.cpu().detach().numpy()))
        with open('decision_transformer_gym_replay/notes/r.txt', 'w') as convert_file:
            convert_file.write(np.array2string(r.cpu().detach().numpy()))
        with open('decision_transformer_gym_replay/notes/rtg.txt', 'w') as convert_file:
            convert_file.write(np.array2string(rtg.cpu().detach().numpy()))
        with open('decision_transformer_gym_replay/notes/timesteps.txt', 'w') as convert_file:
            convert_file.write(np.array2string(timesteps.cpu().detach().numpy()))
        with open('decision_transformer_gym_replay/notes/mask.txt', 'w') as convert_file:
            convert_file.write(np.array2string(mask.cpu().detach().numpy()))
        '''
        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }
 


# Step 5: Extending the Decision Transformer Model to include a loss function
loss_array = []
class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        #print("TrainableDT start forward")
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"] #["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        loss = torch.mean((action_preds - action_targets) ** 2)
        if use_video == True:
            print("loss: ", loss.item())
        #print("TrainableDT end forward")
        loss_array.append(loss.item())
        
        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)


collator = DecisionTransformerGymDataCollator(dataset["train"])
#dataloader = DataLoader(dataset["train"], batch_size=64, num_workers=8)
config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim)
model = TrainableDT(config)

# Step 5.5 optional load checkpoint
if load_checkpoint == True:
    print("load saved checkpoint model")
    #model = AutoModel.from_pretrained(checkpoint_path)
    #model = DecisionTransformerModel.from_pretrained(checkpoint_path) # (model_path)
    #model = DecisionTransformerModel.from_pretrained(checkpoint_path + checkpoint) # (model_path)
    model = pickle.load(open(checkpoint_path + "model_pickle.pkl",'rb')) #works for training
    print("loaded checkpoint")


# Step 6: Defining the training hyperparameters and training the models
time = datetime.datetime.now().strftime('%m-%d_%H:%M')

output_dir = model_path +time+"-"+str(num_train_epochs)+"epochs/"
training_args = TrainingArguments(
    remove_unused_columns=False,
    num_train_epochs=num_train_epochs,
    output_dir=output_dir,
    per_device_train_batch_size=64, #64
    learning_rate=1e-4,   #1e-4,
    weight_decay=1e-4,    #1e-4,
    warmup_ratio=0.1,
    optim="adamw_torch",
    max_grad_norm=0.25,
    dataloader_num_workers=8, ####
    fp16=True, ####
)

print("trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=collator,
)

if train_model == True:
    print("start training")
    trainer.train()  ##

# Step 6.5: Save Model
if save_model == True:
    print("save trained model")
    model.save_pretrained(output_dir)
    #model.save_pretrained(model_path)
    #trainer.save_model(output_dir) #+"save/")

    pickle.dump(model, open(output_dir+"model_pickle.pkl", 'wb'))
    with open(output_dir + 'loss.txt', 'w') as convert_file:
        convert_file.write(str(loss_array))

# Step 7: Visualize the performance of the agent
if test_model == True:

    # home/codysoccerman/miniconda3/envs/imitation/lib/python3.8/site-packages/mujoco_py/
    # Function that gets an action from the model using autoregressive prediction with a window of the previous 20 timesteps.
    def get_action(model, states, actions, rewards, returns_to_go, timesteps):
        # This implementation does not condition on past rewards

        states = states.reshape(1, -1, model.config.state_dim)
        actions = actions.reshape(1, -1, model.config.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:, -model.config.max_length :]
        actions = actions[:, -model.config.max_length :]
        returns_to_go = returns_to_go[:, -model.config.max_length :]
        timesteps = timesteps[:, -model.config.max_length :]
        padding = model.config.max_length - states.shape[1]
        # pad all tokens to sequence length
        attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
        states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
        actions = torch.cat([torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).float()
        returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
        timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

        state_preds, action_preds, return_preds = model.original_forward(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return action_preds[0, -1]

    # build the environment
    device = "cpu" #######
    #accelerator = Accelerator() #####
    #device = accelerator.device ######
    
    color_output = output_dir + "videos_color/"
    os.makedirs(output_dir + "videos_gray/")
    model = model.to(device) ###### model.to("cpu")
    env = gym.make(enviroment)

    env = Recorder(env, color_output, fps=20)
    max_ep_len = 50  #1000   # unsure
    #device = "cpu" #######
    scale = 1  #1000.0  # normalization for rewards/returns
    TARGET_RETURN = 0 / scale #12000 / scale  # evaluation is conditioned on a return of 12000, scaled accordingly

    state_mean = collator.state_mean.astype(np.float16)
    state_std = collator.state_std.astype(np.float16) #32

    act_dim = env.action_space.shape[0]
    state_dim = len(state_mean)#a_len + b_len + c_len #25

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    #print("len(state_mean): ",len(state_mean))
    image_array = []
    gray_image_array = []
    num_videos = 50 # number of episode videos to produce
    low_res = 96 # 96x96 pixels
    for episode in range(num_videos):

        episode_return = 0
        state = env.reset() # fro video we don't use state/typical observation space. instead use pixels

        if use_video == True:
            img = env.render("rgb_array") #rgba #env.render("human") ##
            obs_space_flat, gray_image_array = flatten_image(img, low_res, gray_image_array)

        if use_video == False:
            obs_space_flat = flatten_observation(state)

        
        target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float16).reshape(1, 1) #states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        states = torch.from_numpy(obs_space_flat).reshape(1, state_dim).to(device=device, dtype=torch.float16)  ## [0]
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float16)
        rewards = torch.zeros(0, device=device, dtype=torch.float16)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        
        for t in range(50): #max_ep_len):

            
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = get_action(model, (states - state_mean) / state_std, actions, rewards, target_return, timesteps)
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, _ = env.step(action)
            #print("state after step: ", state)
            if use_video == False:
                obs_space_flat = flatten_observation(state)

            if use_video == True:
                img = env.render("rgb_array") #rgba env.render("human") ##
                obs_space_flat, gray_image_array = flatten_image(img, low_res, gray_image_array)
                
            cur_state = torch.from_numpy(obs_space_flat).to(device=device).reshape(1, state_dim) ## added [0]

            states = torch.cat([states, cur_state], dim=0)
            reward = np.array([reward])
            reward = torch.from_numpy(reward).to(device=device)
            rewards[-1] = reward

            pred_return = target_return[0, -1] - (reward / scale)
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

            episode_return += reward
            
            if done:
                if use_video == True:
                    save_video(image_array, output_dir, "videos_gray/video_{}.mp4".format(episode))
                    image_array = []
                env.reset()
                print("break")
                break
        print("episode: ",episode, " reward: ", episode_return)

    
    print("save videos_color combined video")
    save_video(combined_color_array, output_dir, "videos_color/videos_combined.mp4")

    '''
    print("save videos_gray combined video")
    video_writer = imageio.get_writer(output_dir +"videos_gray/videos_combined.mp4", fps=20)
    for single_image in combined_gray_array:
        video_writer.append_data(np.asarray(single_image))   #single_image))
    video_writer.close()
    '''
else:
    print("test_model == False: Done")


