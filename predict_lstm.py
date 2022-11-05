import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from roboturk_loader_lstm import RoboTurk
from Seq2Vec import Seq2Vec
import PIL
import cv2
import os
import argparse
from utils import Utils

class LSTMPredictor:
    def __init__(self, num_channels, num_kernels, kernel_size, padding, activation, frame_size, num_layers, index, device):
        model = Seq2Vec(num_channels=num_channels, num_kernels=num_kernels, kernel_size=kernel_size, padding=padding, activation=activation, frame_size=frame_size, num_layers=num_layers).to(device)
        model.load_state_dict(torch.load('./checkpoints/model_' + str(index) + '.pt', map_location=torch.device(device)))
        model.eval()
        model = model.to(device)

        self.model = model

    def predict(self, input_sequence, max_length=5):
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # arr = input_sequence.permute(0, 2, 3, 4, 1)
        # whee = arr[0,0,:,:,:]
        # whee = whee*255
        # whee = whee.numpy()
        # whee = whee.astype('uint8')
        # cv2.imshow('whee', whee)
        # cv2.waitKey(5000)

        with torch.no_grad():
            pred = self.model(input_sequence)

        return pred

if __name__ == "__main__":
    frame_size = (64, 64)
    loss_fn = nn.MSELoss()

    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, required=True)
    parser.add_argument('--pred_frames', type=int, default=1) # number of frames to predict
    parser.add_argument('--show', type=bool, default=False)
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--name', type=str, default='default')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = LSTMPredictor(num_channels=3, num_kernels=64, kernel_size=(3,3), padding=(1,1), activation="relu", frame_size=frame_size, num_layers=3, index=args.index, device=device)

    test_dataset = RoboTurk(num_frames=5, stride=1, dir=args.folder, stage='test', shuffle=True, frame_size=frame_size)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            X = batch['data']
            y_actual = batch['y']

            X = X.permute(0, 2, 1, 3, 4)
            X = X.clone().detach().to(device)
            y_actual = y_actual.clone().detach().to(device)

            pred = predictor.predict(X)

            loss = loss_fn(pred, y_actual)

            print('actual:{}\npredicted:{}\nloss:{}'.format(y_actual, pred, loss))
