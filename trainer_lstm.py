import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import math
import numpy as np
from Seq2Vec import Seq2Vec
import torchvision.transforms as transforms
import argparse
import os
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw

from roboturk_loader import RoboTurk

class Trainer():
    def __init__(self, frame_size=(64,64)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)
        self.model = Seq2Vec(num_channels=3, num_kernels=64, kernel_size=(3,3), padding=(1,1), activation="relu", frame_size=frame_size, num_layers=3).to(self.device)

    def train_loop(self, model, opt, loss_fn, dataloader, frames_to_predict): # TODO: move encoding from dataloader to here
        model = model.to(self.device)
        model.train()
        total_loss = 0

        for i, batch in enumerate(tqdm(dataloader)):
            X = batch['data']
            y_actual = batch['y']

            X = X.permute(0, 2, 1, 3, 4)
            X = X.clone().detach().requires_grad_(True).to(self.device)
            # X = torch.tensor(X).to(self.device)
            y_actual = y_actual.clone().detach().to(self.device)

            pred = model(X)

            loss = loss_fn(y_actual, pred)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.detach().item()

        return total_loss / len(dataloader)

    def validation_loop(self, model, loss_fn, dataloader, frames_to_predict):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for j, batch in enumerate(tqdm(dataloader)):
                X = batch['data']
                y_actual = batch['y']

                X = X.permute(0, 2, 1, 3, 4)
                X = X.clone().detach().to(self.device)
                y_actual = y_actual.clone().detach().to(self.device)

                pred = model(X)

                loss = loss_fn(y_actual, pred)

                total_loss += loss.detach().item()

        return total_loss / len(dataloader)

    def fit(self, model, opt, loss_fn, train_dataloader, val_dataloader, epochs, frames_to_predict):
        # Used for plotting later on
        train_loss_list, validation_loss_list = [], []

        print("Training and validating model")
        for epoch in range(epochs):
            if epochs > 1:
                print("-"*25, f"Epoch {epoch + 1}","-"*25)

            train_loss = self.train_loop(model, opt, loss_fn, train_dataloader, frames_to_predict)
            train_loss_list += [train_loss]

            validation_loss = self.validation_loop(model, loss_fn, val_dataloader, frames_to_predict)
            validation_loss_list += [validation_loss]

            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {validation_loss:.4f}")

        # counting number of files in ./checkpoints
        index = len(os.listdir('./checkpoints'))

        if epochs > 1:
            # save model
            torch.save(model.state_dict(), './checkpoints/model' + '_' + str(index) + '.pt')
            print('model saved as model' + '_' + str(index) + '.pt')

        return train_loss_list, validation_loss_list

    def custom_collate(self, batch):
        filtered_batch = []
        for video, _, label in batch:
            filtered_batch.append((video, label))
        return torch.utils.data.dataloader.default_collate(filtered_batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save_best', type=bool, default=False)
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()

    # torch.multiprocessing.set_start_method('spawn')

    frames_per_clip = 5
    frames_to_predict = 5
    stride = 1 # number of frames to shift when loading clips
    batch_size = 32
    epoch_ratio = 1 # to sample just a portion of the dataset
    epochs = 10
    lr = 0.00001
    num_workers = 0
    frame_size = (64, 64)

    dim_model = 256
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout_p = 0.1

    trainer = Trainer(frame_size=frame_size)

    model = trainer.model
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() # TODO: change this to mse + condition + gradient difference
    if args.dataset == 'roboturk':
        train_dataset = RoboTurk(num_frames=5, stride=stride, dir=args.folder, stage='train', shuffle=True, frame_size=frame_size)
        train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=int(len(train_dataset) * epoch_ratio))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=num_workers)

        test_dataset = RoboTurk(num_frames=5, stride=stride, dir=args.folder, stage='test', shuffle=True, frame_size=frame_size)
        test_sampler = RandomSampler(test_dataset, replacement=False, num_samples=int(len(test_dataset) * epoch_ratio))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=num_workers)

    if args.save_best:
        best_loss = 1e10
        epoch = 1
        while True:
            print("-"*25, f"Epoch {epoch}","-"*25)
            train_loss_list, validation_loss_list = trainer.fit(model=model, opt=opt, loss_fn=loss_fn, train_dataloader=train_loader, val_dataloader=test_loader, epochs=1, frames_to_predict=frames_to_predict)
            if validation_loss_list[-1] < best_loss:
                best_loss = validation_loss_list[-1]
                torch.save(model.state_dict(), './checkpoints/model_' + args.name + '.pt')
                print('model saved as model_' + str(args.name) + '.pt')
            epoch += 1
    else:
        trainer.fit(model=model, opt=opt, loss_fn=loss_fn, train_dataloader=train_loader, val_dataloader=test_loader, epochs=epochs, frames_to_predict=frames_to_predict)