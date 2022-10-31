import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import math
import numpy as np
from transformer import Transformer
import torchvision.transforms as transforms
import argparse
import os
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw

from roboturk_loader_cnn import RoboTurk

class Trainer():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)
        self.resnet50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        # freeze resnet50
        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.resnet50.fc = nn.Linear(2048, 8)
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True
        self.model = self.resnet50

    def encode_img(self, img):
        # turn an image into image latents
        # input image into CNN
        # reshape img to 224x224
        img = cv2.resize(img, (224, 224))
        # img = img.reshape((1, 224, 224, 3))
        latents = self.resnet50(img)
        return latents


    def train_loop(self, model, opt, loss_fn, dataloader, frames_to_predict): # TODO: move encoding from dataloader to here
        model = model.to(self.device)
        model.train()
        total_loss = 0

        for i, batch in enumerate(tqdm(dataloader)):
            X = batch['data']
            y = batch['y']

            X = X.clone().detach().to(self.device)
            y = y.clone().detach().to(self.device)
            # X = torch.tensor(X).to(self.device)
            # y = torch.tensor(y).to(self.device)

            pred = model(X)
            loss = loss_fn(pred, y)

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
                y = batch['y']

                X = X.clone().detach().to(self.device)
                y = y.clone().detach().to(self.device)
                # X = torch.tensor(X).to(self.device)
                # y = torch.tensor(y).to(self.device)

                pred = model(X)
                loss = loss_fn(pred, y)

                pred = model(X)
                loss = loss_fn(pred, y)

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
    parser.add_argument('--save_best', action='store_true')
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
    lr = 0.001
    num_workers = 0

    dim_model = 256
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout_p = 0.1

    trainer = Trainer()

    model = trainer.model
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() # TODO: change this to mse + condition + gradient difference
#                                               collate_fn=trainer.custom_collate)
    if args.dataset == 'roboturk':
        train_dataset = RoboTurk(num_frames=5, stride=stride, dir=args.folder, stage='train', shuffle=True)
        train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=int(len(train_dataset) * epoch_ratio))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=num_workers)

        test_dataset = RoboTurk(num_frames=5, stride=stride, dir=args.folder, stage='test', shuffle=True)
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
