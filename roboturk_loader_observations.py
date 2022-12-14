import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from transformer import Transformer
import torchvision.transforms as transforms
import argparse
import cv2
import os
import glob
import copy
from torch.utils.data import DataLoader, RandomSampler

class RoboTurkObs(data.Dataset):
    def __init__(self, num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='raw', shuffle=True):
        self.stage = stage
        self.dir = os.path.join(dir, stage)
        self.num_frames = num_frames
        self.stride = stride
        self.indices, self.dataset = self.get_data(shuffle=shuffle)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = Transformer()
        # self.SOS_token = torch.ones((1, model.dim_model), dtype=torch.float32, device=device) * 2
        # self.EOS_token = torch.ones((1, model.dim_model), dtype=torch.float32) * 3

        test_X = np.concatenate((np.arange(0, 26) + 100, np.arange(0, 26) + 900))
        test_y = np.concatenate((np.arange(22, 26) + 100, np.arange(22, 26) + 900))
        self.mean_x = np.mean(test_X)
        self.stdev_x = np.std(test_X)
        self.mean_y = np.mean(test_y)
        self.stdev_y = np.std(test_y)

    def __getitem__(self, index):
        # obtaining file paths
        obs_names = self.dataset[index][0]
        act_names = self.dataset[index][1]



        # loading and formatting image
        frames = []
        # this is for loading observation spaces
        # np.random.seed(0)
        for obs in obs_names:
            if (obs == 0):
                dat = np.zeros(25)
            else:
                dat = np.load(obs)
            # dat = np.load(obs, allow_pickle=True).item()
            # dat = np.concatenate([x.flatten() for x in dat.values()])
            # dat = np.arange(0, 26) + np.random.randint(0, 1000) # testing
            # dat = np.arange(0, 26) + np.random.choice(np.array([100,900])) # testing
            # dat = (dat - self.mean_x)/self.stdev_x
            dat = torch.from_numpy(dat)
            dat = dat.float()

            frames.append(dat)

        # this was for loading action spaces
        # for frame in obs_names:
        #     frame = np.load(frame, allow_pickle=True)
        #     # frame = torch.from_numpy(frame)
        #     frame = torch.tensor(frame)
        #     frame = frame.float()
        #     frame = frame.flatten()
        #     frames.append(frame)

        frames = torch.stack(frames, dim=0)
        frames = frames.detach()

        # frames = torch.diff(frames, dim=0)
        frames.requires_grad = False

        joints = np.load(act_names)
        # joints = torch.mean(frames, dim=0)[-4:]
        # joints = (joints - self.mean_y)/self.stdev_y
        # joints = torch.from_numpy(joints)
        joints = torch.tensor(joints)
        joints = joints.float()
        joints = joints.flatten()

        # # concatenating SOS token,
        # frames = torch.cat((self.SOS_token, frames), dim=0)

        #  frames.shape: (seq_len + 1, dim_model)
        return {'data':frames, 'y':joints}

    def __len__(self):
        return len(self.dataset)

    def get_data(self, shuffle):
        obs_names = []
        act_names = []
        dataset = []
        indices = []

        # crawling the directory
        # for dir in glob.glob(os.path.join(self.dir, '*/'), recursive=True):
        #     parent = os.path.split(os.path.split(dir)[0])[1]
        #     for file in glob.glob(os.path.join(dir, '*.jpg')):
        #             parent_index = parent.split('_')[-1]
        #             if parent_index != 'depth': # TODO: change this if we add depth
        #                 obs_names.append((int(parent_index+file[-7:-4]), os.path.join(dir, file)))
        #     for file in glob.glob(os.path.join(dir, '*.npy')):
        #         parent_index = parent.split('_')[-2]
        #         act_names.append((int(parent_index+file[-7:-4]), os.path.join(dir, file)))

        for dir, _, files in os.walk(self.dir):
            if (len(files) == 0):
                continue
            for file in files:
                parent = dir.split('/')[-1]
                # (parent+index, name)
                if ('observations' in parent):
                    parent_index = parent.split('_')[-1]
                    obs_names.append((int(parent_index+file[-9:-4]), os.path.join(dir, file)))
                if ('actions' in parent):
                    parent_index = parent.split('_')[-1]
                    act_names.append((int(parent_index+file[-9:-4]), os.path.join(dir, file)))

        # sorting the names numerically. first 4 digits are folder and last 3 are file
        obs_names = sorted(obs_names, key=lambda x: x[0])
        act_names = sorted(act_names, key=lambda x: x[0])

        # indices = [x[0] for x in obs_names]

        # # for i in range(0, len(obs_namesobs_anmes), self.num_frames): # for each sequence of frames
        # for i in range(0, len(obs_names) - self.num_frames, self.num_frames): # for each sequence
        #     index_list = []
        #     frame_names = []
        #     joint_frame_names = []
        #     # for j in range(0, self.stride*(self.num_frames - 1) + 1, self.stride): # for each frame in the sequence
        #     for j in range(self.num_frames):
        #         index_list.append(obs_names[i+j][0]) # getting frame i, i+self.stride, i+2*self.stride, ...
        #         frame_names.append(obs_names[i+j][1])
        #         joint_frame_names.append(act_names[i+j][1])

        for i in range(0, len(obs_names) - self.num_frames * self.stride - 1):
            index_list = []
            frame_names = []
            for j in range(self.stride): # don't miss the skipped frames from the stride
                if i % self.stride == j:
                    if (str(obs_names[i][0])[-5:] == '00000'):
                        self.append_sos(dataset, indices, obs_names, act_names, i)

                    for k in range(self.num_frames): # for each sequence
                        index_list.append(obs_names[i+k*self.stride][0]) # getting frame i, i+self.stride, i+2*self.stride, ... (i+1)+self.stride, (i+1)+2*self.stride, ... etc
                        frame_names.append(obs_names[i+k*self.stride][1])

                    if (not np.all(np.diff(index_list) == self.stride)):
                        # frames arent contiguous
                        # we cant use the last sequence in a video because we need a label for the seq+1 action
                        continue

                    # list of lists of frame indices
                    indices.append(index_list)

                    # each element is a list of frame names with length num_frames and skipping frames according to stride
                    dataset.append((frame_names, act_names[i+k*self.stride][1]))

                    # print('frame_names: ', frame_names)

        if shuffle:
            np.random.shuffle(dataset)
        else:
            dataset = np.array(dataset)

        return indices, dataset

    def append_sos(self, dataset, indices, obs_names, act_names, ind):
        # add sequences to the dataset with zero tokens before the start of the solve
        # ind: index of actual start of sequence
        for i in range(0, self.num_frames - 1):
            index_list = [0]*(self.num_frames - i - 1)
            frame_names = [0]*(self.num_frames - i - 1)
            pad_len = len(frame_names)
            for j in range(0, self.num_frames - pad_len):
                index_list.append(obs_names[ind+j][0])
                frame_names.append(obs_names[ind+j][1])
            act_name = act_names[ind+j][1]

            dataset.append((frame_names, act_name))
            indices.append(index_list)

if __name__ == '__main__':
    dataset = RoboTurkObs(num_frames=5, stride=1, dir='data/PandaPickAndPlace-v1/data', stage='train', shuffle=True)
    # dataset = RoboTurk(num_frames=5, stride=1, dir='/media/jer/Crucial X6/data/RoboTurk_videos/bins-Bread', stage='train', shuffle=True)
    # test_sampler = RandomSampler(dataset, replacement=False, num_samples=int(len(dataset) * 0.01))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # joints = []
    # for i, data in enumerate(test_loader):
    #     joint = dataset[i]['y']
    #     joints.append(joint)
    # joints = torch.cat(joints, dim=0)
    # print('joints shape: ', torch.tensor(joints).shape)
    # print('avg joints: ', torch.mean(joints, dim=0))

    print(dataset)


    for i in range(10):
        print('dir: ', dataset.dir)
        print('clip ', i)
        print("clips in the dataset: ", len(dataset.dataset))
        # print('clip length: ', len(dataset[0]))
        print('dataset: ', dataset[i])
        print('frame shape: ', dataset[i]['data'].shape)
        print('joint shape: ', dataset[i]['y'].shape)
        frames = dataset[i]['data']
        jointdata = dataset[i]['y']
