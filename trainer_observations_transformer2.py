import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from roboturk_loader_observations import RoboTurkObs
from transformer import Transformer
from lstm import ShallowRegressionLSTM
import argparse

frames_per_clip = 2
stride = 1
epoch_ratio = 1

batch_size = 32
lr = 1e-3
num_workers = 6
epochs = 30

model = ShallowRegressionLSTM(input_size=26, output_size=4, hidden_units=2048, num_layers=8)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

train_dataset = RoboTurkObs(num_frames=frames_per_clip, stride=stride, dir='/home/alanhesu/Documents/github/vroom/pandapickandplace/data', stage='train', shuffle=True)
train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=int(len(train_dataset) * epoch_ratio))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=num_workers)

test_dataset = RoboTurkObs(num_frames=frames_per_clip, stride=stride, dir='/home/alanhesu/Documents/github/vroom/pandapickandplace/data', stage='test', shuffle=True)
test_sampler = RandomSampler(test_dataset, replacement=False, num_samples=int(len(test_dataset) * epoch_ratio))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=num_workers)

def train_epoch(epoch_index):
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(tqdm(train_loader)):
        X = data['data']
        y_expected = data['y']

        optimizer.zero_grad()

        pred = model(X)

        loss = loss_fn(pred, y_expected)
        loss.backward()

        optimizer.step()

        print('input', X[0,:,:])
        print('expected', y_expected[0,:])
        print('predicted', pred[0,:])

        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss/100
            print(' batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index*len(train_loader) + i + 1
            running_loss = 0

    return running_loss

epoch_number = 0

for epoch in range(epochs):
    print('epoch {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss = train_epoch(epoch_number)

    model.train(False)

    running_vloss = 0
    for i, vdata in enumerate(tqdm(test_loader)):
        X = vdata['data']
        y_expected = vdata['y']
        pred = model(X)
        vloss = loss_fn(pred, y_expected)
        running_vloss += vloss

        # print('input', X[0,:,:])
        # print('expected', y_expected[0,:])
        # print('predicted', pred[0,:])

    avg_vloss = running_vloss/(i+1)
    print('loss train {} valid {}'.format(avg_loss, avg_vloss))

    epoch_number += 1

# parser = argparse.ArgumentParser()
# parser.add_argument('--folder', type=str, required=True) # dataset location

# args = parser.parse_args()

