import numpy as np
import imageio
import os
import pickle
import shutil
import fnmatch
import skvideo.io
import json
#from imitation.data.types import TrajectoryWithRew

'''
cd /home/codysoccerman/Documents/classes/Fall_22/Deep_Learning/Project/rl-baselines3-zoo-master
conda activate imitation
python3 create_datastructure.py

'''
using_videos = False
save_data = True
delete_old_data = False
behavioral_cloning = False
dec_transformer = False
vpt_actions = True
n=1000


dicts = {}
dict_list = []
obs_list = []
acts_list = []
rews_list = []
reward_sum = 0
env_name =  "PandaPickAndPlace-v1" #"PandaReach-v1"    #"PandaPickAndPlace-v1"
recording_path = "recording_data/" + env_name

if delete_old_data == True and behavioral_cloning == False:
    #home/codysoccerman/.cache/huggingface/datasets/decision_transformer_gym_replay/PandaPickAndPlace-v1/
    if os.path.exists("/home/codysoccerman/.cache/huggingface/datasets/decision_transformer_gym_replay/" + env_name + "/1.1.0"): #/datasets/decision_transformer_gym_replay/" + env_name):
        print("exists")
        shutil.rmtree("/home/codysoccerman/.cache/huggingface/datasets/decision_transformer_gym_replay/" + env_name + "/1.1.0")      #PandaPickAndPlace-v1/1.1.0")
        print("old files deleted")
    else:
        print("no files to delete")

# videos (choose type)
#video_type = "/videos/"
#video_type = "/videos_low_res/"
video_type = "/videos_grayscaled/"
low_res = 96 #96x96

# no videos
if using_videos == False and behavioral_cloning == True: 
    print("creating BC datastructure")
    for idx in range(1,n+1):
        #print(idx)
        observations_array = np.load(recording_path + '/observations/observations_{}.npy'.format(idx), allow_pickle=True)
        action_array = np.load(recording_path + '/actions/actions_{}.npy'.format(idx), allow_pickle=True)
        rewards_array = np.load(recording_path + '/rewards/rewards_{}.npy'.format(idx), allow_pickle=True)

        dones_array = np.load(recording_path + '/dones/dones_{}.npy'.format(idx), allow_pickle=True)
        traj = TrajectoryWithRew(obs=np.asarray(observations_array), acts=np.asarray(action_array[:-1]),infos=None, terminal=True, rews=np.asarray(np.ones(len(action_array[:-1]))))    
        dict_list.append(traj)


if using_videos == False and dec_transformer == True: 
    print("creating non video datastructure")
    for idx in range(1,n+1):
        #print(idx)
        observations_array = np.load(recording_path + '/observations/observations_{}.npy'.format(idx), allow_pickle=True)
        action_array = np.load(recording_path + '/actions/actions_{}.npy'.format(idx), allow_pickle=True)
        rewards_array = np.load(recording_path + '/rewards/rewards_{}.npy'.format(idx), allow_pickle=True)
        dones_array = np.load(recording_path + '/dones/dones_{}.npy'.format(idx), allow_pickle=True)
    
        dicts = {"observations": observations_array, "acts": action_array, "rewards": rewards_array, "dones": dones_array}
        dict_list.append(dicts)


if vpt_actions == True: 
    print("creating vpt_actions non video datastructure")
    for idx in range(1,n+1):
        #print(idx)
        #observations_array = np.load(recording_path + '/observations/observations_{}.npy'.format(idx), allow_pickle=True)
        action_array = np.load(recording_path + '/actions/actions_{}.npy'.format(idx), allow_pickle=True)
        action_array = action_array.tolist()
        #rewards_array = np.load(recording_path + '/rewards/rewards_{}.npy'.format(idx), allow_pickle=True)
        #dones_array = np.load(recording_path + '/dones/dones_{}.npy'.format(idx), allow_pickle=True)
    
        #dicts = {"observations": observations_array, "acts": action_array, "rewards": rewards_array, "dones": dones_array}
        #dict_list.append(dicts)
        if save_data == True:
            with open('vpt_bc/data/PandaPickAndPlace-v1/video_{}.jsonl'.format(idx), 'w') as f:
                for action in action_array:
                    dicts = {"actions": action}
                    f.write(json.dumps(dicts) + "\n")
    print("data saved")
########################################################

# videos
if using_videos == True and dec_transformer == True: 
    print("creating video datastructure")
    for idx in range(1,n+1):
        print(idx)

        # Load video
        video = skvideo.io.vread(recording_path + video_type + 'video_{}.mp4'.format(idx))
        image_array = np.asarray(video, dtype='float16')
        image_array = image_array[:,:,:,1] 
        observations_array=np.reshape(image_array,(50,low_res*low_res))/255
         
        action_array = np.load(recording_path + '/actions/actions_{}.npy'.format(idx), allow_pickle=True)

        rewards_array = np.load(recording_path + '/rewards/rewards_{}.npy'.format(idx), allow_pickle=True)

        dones_array = np.load(recording_path + '/dones/dones_{}.npy'.format(idx), allow_pickle=True)
         
        dicts = {"observations": observations_array, "actions": action_array, "rewards": rewards_array, "dones": dones_array}
        dict_list.append(dicts)


if using_videos == True and behavioral_cloning == True: 
    print("creating video datastructure")
    for idx in range(1,n+1):
        print(idx)

        # Load video
        video = skvideo.io.vread(recording_path + video_type + 'video_{}.mp4'.format(idx))
        image_array = np.asarray(video, dtype=np.float16) #'float16')
        image_array = image_array[:,:,:,1] 
        observations_array=np.reshape(image_array,(50,low_res*low_res))/255
         
        action_array = np.load(recording_path + '/actions/actions_{}.npy'.format(idx), allow_pickle=True)

        rewards_array = np.load(recording_path + '/rewards/rewards_{}.npy'.format(idx), allow_pickle=True)

        dones_array = np.load(recording_path + '/dones/dones_{}.npy'.format(idx), allow_pickle=True)
         
        dones_array = np.load(recording_path + '/dones/dones_{}.npy'.format(idx), allow_pickle=True)

        traj = TrajectoryWithRew(obs=np.asarray(observations_array), acts=np.asarray(action_array[:-1]),infos=None, terminal=True, rews=np.asarray(np.ones(len(action_array[:-1]))))    
        dict_list.append(traj)


#TrajectoryWithRew_list = TrajectoryWithRew(obs=np.asarray(obs_list), acts=np.asarray(acts_list[:-1]),infos=None, terminal=True, rews=np.asarray(np.ones(len(acts_list[:-1]))))      
#print(TrajectoryWithRew_list)
# save data

if save_data == True and behavioral_cloning == True:
    with open("my_models/behavioral_cloning/" + env_name + ".pkl", 'wb') as f:
        pickle.dump(dict_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("data saved")


if save_data == True and dec_transformer == True:
    with open("decision_transformer_gym_replay/" + env_name + ".pkl", 'wb') as f:
        pickle.dump(dict_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("data saved")

