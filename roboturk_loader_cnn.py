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

class RoboTurk(data.Dataset):
    def __init__(self, num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='raw', shuffle=True):
        self.stage = stage
        self.dir = os.path.join(dir, stage)
        self.num_frames = num_frames
        self.stride = stride
        self.dataset = self.get_data(shuffle=shuffle)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, index):
        # obtaining file paths
        img_name = self.dataset[index][0]
        jointdata_name = self.dataset[index][1]

        frame = cv2.imread(img_name)
        frame = cv2.resize(frame, (224, 224))
        frame = torch.from_numpy(frame)
        frame = frame.permute(2, 0, 1)
        frame = frame.float()/255.0

        jointdata = np.load(jointdata_name)
        jointdata = torch.from_numpy(jointdata)
        jointdata = jointdata.float()

        return {'data': frame, 'y': jointdata}
        # frame_names = self.dataset[index]

        # loading and formatting image
        frames=[]
        for frame_name in frame_names:
            frame = cv2.imread(frame_name)
            frame = cv2.resize(frame, (224, 224))
            # frame = self.transform(frame) # TODO: add transforms

        #     # check decoding
        #     # reconstruction = self.sd_utils.decode_img_latents(frame)
        #     # reconstruction = np.array(reconstruction[0])
        #     # cv2.imshow('reconstruction', reconstruction)
        #     # cv2.waitKey(0)

            frame = frame.squeeze(0)
        #     # frame = torch.from_numpy(frame)
        #     # frame = frame.permute(2, 0, 1)
        #     # frame = frame.float()/255.0

        #     # bs, seq_len, _, _, _ = frame.shape
            frame = frame.flatten()

            frames.append(frame)

        # # creating tensor with requires_grad=False
        frames = torch.stack(frames, dim=0)
        frames = frames.detach()
        frames.requires_grad = False

        # # concatenating SOS token,
        frames = torch.cat((self.SOS_token, frames), dim=0)

        #  frames.shape: (seq_len + 1, dim_model)
        return self.indices[index], frames

    def __len__(self):
        return len(self.dataset)

    def get_data(self, shuffle):
        img_names = []
        joint_data = []
        dataset = []
        indices = []

        # crawling the directory
        for dir in glob.glob(os.path.join(self.dir, '*/'), recursive=True):
            parent = os.path.split(os.path.split(dir)[0])[1]
            for file in glob.glob(os.path.join(dir, '*.jpg')):
                parent_index = parent.split('_')[-1]
                img_names.append((int(parent_index), os.path.join(dir, file)))
            for file in glob.glob(os.path.join(dir, '*.npy')):
                parent_index = parent.split('_')[-2]
                joint_data.append((int(parent_index), os.path.join(dir, file)))

        # sorting the names numerically. first 4 digits are folder and last 3 are file
        img_names = sorted(img_names, key=lambda x: x[0])
        joint_data = sorted(joint_data, key=lambda x: x[0])

        for i in range(0, len(img_names)):
            dataset.append((img_names[i][1], joint_data[i][1]))

        # indices = [x[0] for x in img_names]

        # for i in range(0, len(img_names), self.num_frames):
            # index_list = []
            # frame_names = []
            # for j in range(0, self.num_frames, self.stride):
                # if i+j < len(img_names):
                    # index_list.append(img_names[i+j][0])
                    # frame_names.append(img_names[i+j][1])
                # else:
                    # break

            # # list of lists of frame indices
            # indices.append(index_list)

            # # each element is a list of frame names with length num_frames and skipping frames according to stride
            # dataset.append(frame_names, joint_data[i][1])

        # if shuffle:
            # np.random.shuffle(dataset)
        # else:
            # dataset = np.array(dataset)

        return dataset


if __name__ == '__main__':
    dataset = RoboTurk(num_frames=5, stride=1, dir='/mnt/batch/tasks/shared/LS_root/mounts/clusters/ahesu62/code/Users/ahesu6/github/vroom/RoboTurk_videos/bins-Bread', stage='train', shuffle=True)

    print(len(dataset.datset))
