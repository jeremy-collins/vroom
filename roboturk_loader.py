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
from torch.utils.data import DataLoader, RandomSampler

class RoboTurk(data.Dataset):
    def __init__(self, num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='raw', shuffle=True, frame_size=(224,224)):
        self.stage = stage
        self.dir = os.path.join(dir, stage)
        self.num_frames = num_frames
        self.stride = stride
        self.frame_size = frame_size
        self.indices, self.dataset = self.get_data(shuffle=shuffle)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = Transformer()
        # self.SOS_token = torch.ones((1, model.dim_model), dtype=torch.float32, device=device) * 2
        # self.EOS_token = torch.ones((1, model.dim_model), dtype=torch.float32) * 3

    def __getitem__(self, index):
        # obtaining file paths
        frame_names = self.dataset[index][0]
        joint_names = self.dataset[index][1]

        

        # loading and formatting image
        frames=[]
        for frame_name in frame_names:
            frame = cv2.imread(frame_name)
            frame = cv2.resize(frame, self.frame_size)
            # frame = self.transform(frame) # TODO: add transforms

            # # view image
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)

            # frame = frame.squeeze(0)
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)
            frame = frame.float()/255.0

        #     # bs, seq_len, _, _, _ = frame.shape
            # frame = frame.flatten()

            frames.append(frame)

        frames = torch.stack(frames, dim=0)
        frames = frames.detach()
        frames.requires_grad = False

        jointdata=[]
        for joint_name in joint_names:
            joints = np.load(joint_name)
            joints = torch.from_numpy(joints)
            joints = joints.float()
            jointdata.append(joints)

        jointdata = torch.stack(jointdata, dim=0) # bs, seq_len, 8
            # jointdata = np.load(self.dataset[index][1])
            # jointdata = torch.from_numpy(jointdata).float()

        # # concatenating SOS token,
        # frames = torch.cat((self.SOS_token, frames), dim=0)

        #  frames.shape: (seq_len + 1, dim_model)
        return {'data':frames, 'y':jointdata}

    def __len__(self):
        return len(self.dataset)

    def get_data(self, shuffle):
        img_names = []
        joint_names = []
        dataset = []
        indices = []

        # crawling the directory
        # for dir in glob.glob(os.path.join(self.dir, '*/'), recursive=True):
        #     parent = os.path.split(os.path.split(dir)[0])[1]
        #     for file in glob.glob(os.path.join(dir, '*.jpg')):
        #             parent_index = parent.split('_')[-1]
        #             if parent_index != 'depth': # TODO: change this if we add depth
        #                 img_names.append((int(parent_index+file[-7:-4]), os.path.join(dir, file)))
        #     for file in glob.glob(os.path.join(dir, '*.npy')):
        #         parent_index = parent.split('_')[-2]
        #         joint_names.append((int(parent_index+file[-7:-4]), os.path.join(dir, file)))

        for dir, _, files in os.walk(self.dir):
            for file in files:
                parent = dir.split('/')[-1]
                # (parent+index, name)
                if file.endswith('.jpg'):
                    parent_index = parent.split('_')[-1]
                    if parent_index != 'depth': # TODO: change this if we add depth
                        img_names.append((int(parent_index+file[-8:-4]), os.path.join(dir, file)))
                if file.endswith('.npy'):
                    parent_index = parent.split('_')[-2]
                    joint_names.append((int(parent_index+file[-8:-4]), os.path.join(dir, file)))

        # sorting the names numerically. first 4 digits are folder and last 3 are file
        img_names = sorted(img_names, key=lambda x: x[0])
        joint_names = sorted(joint_names, key=lambda x: x[0])

        # indices = [x[0] for x in img_names]

        # # for i in range(0, len(img_names), self.num_frames): # for each sequence of frames
        # for i in range(0, len(img_names) - self.num_frames, self.num_frames): # for each sequence
        #     index_list = []
        #     frame_names = []
        #     joint_frame_names = []
        #     # for j in range(0, self.stride*(self.num_frames - 1) + 1, self.stride): # for each frame in the sequence
        #     for j in range(self.num_frames):
        #         index_list.append(img_names[i+j][0]) # getting frame i, i+self.stride, i+2*self.stride, ...
        #         frame_names.append(img_names[i+j][1])
        #         joint_frame_names.append(joint_names[i+j][1])

        for i in range(0, len(img_names) - self.num_frames * self.stride):
            index_list = []
            frame_names = []
            joint_frame_names = []
            for j in range(self.stride): # don't miss the skipped frames from the stride
                if i % self.stride == j:
                    for k in range(self.num_frames): # for each sequence
                        index_list.append(img_names[i+k*self.stride][0]) # getting frame i, i+self.stride, i+2*self.stride, ... (i+1)+self.stride, (i+1)+2*self.stride, ... etc
                        frame_names.append(img_names[i+k*self.stride][1])
                        joint_frame_names.append(joint_names[i+k*self.stride][1])

                    # list of lists of frame indices
                    indices.append(index_list)

                    # each element is a list of frame names with length num_frames and skipping frames according to stride
                    dataset.append((frame_names, joint_frame_names))

                    # print('frame_names: ', frame_names)

        if shuffle:
            np.random.shuffle(dataset)
        else:
            dataset = np.array(dataset)

        return indices, dataset


if __name__ == '__main__':
    dataset = RoboTurk(num_frames=5, stride=15, dir='data/RoboTurk_videos/bins-Bread', stage='train', shuffle=True)
    # dataset = RoboTurk(num_frames=5, stride=1, dir='/media/jer/Crucial X6/data/RoboTurk_videos/bins-Bread', stage='train', shuffle=True)
    test_sampler = RandomSampler(dataset, replacement=False, num_samples=int(len(dataset) * 0.01))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=test_sampler, num_workers=0)

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
        for i, frame in enumerate(frames):
            print(frame.size())
            print('joint data: ', jointdata[i])
            frame = frame.permute(1, 2, 0)
            cv2.imshow('frame', np.array(frame))
            cv2.waitKey(0)