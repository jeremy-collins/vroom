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

class Panda(data.Dataset):
    def __init__(self, num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='raw', shuffle=True, frame_size=(224, 224), stack=True):
        self.stage = stage
        self.dir = os.path.join(dir, stage)
        self.num_frames = num_frames
        self.stride = stride
        self.frame_size = frame_size
        self.indices, self.dataset = self.get_data(shuffle=shuffle)
        self.stack = stack
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = Transformer()
        # self.SOS_token = torch.ones((1, model.dim_model), dtype=torch.float32, device=device) * 2
        # self.EOS_token = torch.ones((1, model.dim_model), dtype=torch.float32) * 3

    def __getitem__(self, index):
        # obtaining file paths
        frame_names = self.dataset[index][0]
        act_names = self.dataset[index][1]

        # loading and formatting image
        frames=[]
        for frame_name in frame_names:
            if (frame_name == 0):
                frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
            else:
                frame = cv2.imread(frame_name)
                frame = cv2.resize(frame, self.frame_size)
            # frame = self.transform(frame) # TODO: add transforms

        #     # check decoding
        #     # reconstruction = self.sd_utils.decode_img_latents(frame)
        #     # reconstruction = np.array(reconstruction[0])
        #     # cv2.imshow('reconstruction', reconstruction)
        #     # cv2.waitKey(0)

            # frame = frame.squeeze(0)
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)
            frame = frame.float()/255.0

        #     # bs, seq_len, _, _, _ = frame.shape
            # frame = frame.flatten()

            frames.append(frame)

        # # creating tensor with requires_grad=False
        if (self.stack): # a hack so we can toggle between sequences and single frames
            frames = torch.stack(frames, dim=0)
        else:
            frames = frames[0]
        frames = frames.detach()
        frames.requires_grad = False

        joints = np.load(act_names)
        joints = torch.tensor(joints)
        joints = joints.float()
        joints = joints.flatten()

        return {'data':frames, 'y':joints}

    def __len__(self):
        return len(self.dataset)

    def get_data(self, shuffle):
        img_names = []
        act_names = []
        dataset = []
        indices = []

        # crawling the directory
        for dir, _, files in os.walk(self.dir):
            if (len(files) == 0):
                continue
            for file in files:
                parent = dir.split('/')[-1]
                # (parent+index, name)
                if ('video' in parent):
                    parent_index = parent.split('_')[-1]
                    img_names.append((int(parent_index+file[-8:-4]), os.path.join(dir, file)))
                if ('actions' in parent):
                    parent_index = parent.split('_')[-1]
                    act_names.append((int(parent_index+file[-8:-4]), os.path.join(dir, file)))

        # sorting the names numerically. first 4 digits are folder and last 3 are file
        # img_ind = [x[0] for x in img_names]
        # act_ind = [x[0] for x in act_names]
        # for i in img_ind:
        #     if i not in act_ind:
        #         print(i)
        img_names = sorted(img_names, key=lambda x: x[0])
        act_names = sorted(act_names, key=lambda x: x[0])

        # indices = [x[0] for x in img_names]

        for i in range(0, len(img_names) - self.num_frames * self.stride - 1):
            index_list = []
            frame_names = []
            for j in range(self.stride): # don't miss the skipped frames from the stride
                if i % self.stride == j:
                    if (str(img_names[i][0])[-4:] == '0000'):
                        self.append_sos(dataset, indices, img_names, act_names, i)

                    for k in range(self.num_frames): # for each sequence
                        index_list.append(img_names[i+k*self.stride][0]) # getting frame i, i+self.stride, i+2*self.stride, ... (i+1)+self.stride, (i+1)+2*self.stride, ... etc
                        frame_names.append(img_names[i+k*self.stride][1])

                    if (not np.all(np.diff(index_list) == 1)):
                        # frames arent contiguous
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

    def append_sos(self, dataset, indices, img_names, act_names, ind):
        # add sequences to the dataset with zero tokens before the start of the solve
        # ind: index of actual start of sequence
        for i in range(0, self.num_frames - 1):
            index_list = [0]*(self.num_frames - i - 1)
            frame_names = [0]*(self.num_frames - i - 1)
            pad_len = len(frame_names)
            for j in range(0, self.num_frames - pad_len):
                index_list.append(img_names[ind+j][0])
                frame_names.append(img_names[ind+j][1])
            act_name = act_names[ind+j][1]

            dataset.append((frame_names, act_name))
            indices.append(index_list)

if __name__ == '__main__':
    dataset = Panda(num_frames=5, stride=1, dir='data/PandaPickAndPlace-v1/data', stage='train', shuffle=True)

    for i in range(10):
        print('dir: ', dataset.dir)
        print('clip ', i)
        print("clips in the dataset: ", len(dataset.dataset))
        # print('clip length: ', len(dataset[0]))
        print('frame shape: ', dataset[0]['data'].shape)
        frames = dataset[i]['data']
        for frame in frames:
            print(frame.size())
            frame = frame.permute(1, 2, 0)
            cv2.imshow('frame', np.array(frame))
            cv2.waitKey(0)
