import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import math
import numpy as np
from transformer_observations import TransformerObs
import torchvision.transforms as transforms
import argparse
import os
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw
from utils import Utils

from roboturk_loader_observations import RoboTurkObs

class TrainerObs():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)
        # self.utils = Utils()
        # self.utils.init_resnet()
        # self.utils.init_resnet(freeze=False)

    # def encode_img(self, img):
    #     # input image into CNN
    #     # img = np.array(img, dtype=np.float32)
    #     # img = cv2.resize(img, (224, 224))
    #     latents = self.resnet50(img)
    #     # img = torch.tensor(latents).to(self.device)
    #     return latents


    def train_loop(self, model, opt, loss_fn, dataloader, frames_to_predict):
        model = model.to(self.device)
        model.train()
        total_loss = 0

        for i, batch in enumerate(tqdm(dataloader)):
            X = batch['data']
            # X = torch.tensor(X).to(self.device)
            X = X.clone().detach().requires_grad_(True).to(self.device)

            y = batch['y']
            y = y.clone().detach().to(self.device)
            # y = torch.tensor(y).to(self.device)

            pred = model(X)

            y_expected = batch['y']



            y_expected = torch.tensor(y_expected).to(self.device)
            y_expected = y_expected.permute(1, 0, 2)

            # model.out = model_dim -> 8, to compare with ground truth
            # pred is sequence of next projected embeddings, y_expected is sequence of ground truth joint velocities
            # loss = loss_fn(pred[-frames_to_predict:], y_expected[-frames_to_predict:])
            loss = loss_fn(pred, y_expected[-1])

            # print(pred[-frames_to_predict:].shape, y_expected[-frames_to_predict:].shape)
            # print(pred[-frames_to_predict:, 0], y_expected[-frames_to_predict:, 0])

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.detach().item()

        return total_loss / len(dataloader)

    def validation_loop(self, model, loss_fn, dataloader, frames_to_predict):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                X = batch['data']
                # X = torch.tensor(X).to(self.device)
                X = X.clone().detach().requires_grad_(True).to(self.device)

                y = batch['y']
                # y = torch.tensor(y).to(self.device)
                y = y.clone().detach().to(self.device)

                pred = model(X)

                y_expected = batch['y']



                y_expected = torch.tensor(y_expected).to(self.device)
                y_expected = y_expected.permute(1, 0, 2)

                # model.out = model_dim -> 8, to compare with ground truth
                # pred is sequence of next projected embeddings, y_expected is sequence of ground truth joint velocities
                # loss = loss_fn(pred[-frames_to_predict:], y_expected[-frames_to_predict:])
                loss = loss_fn(pred, y_expected[-1])

                # print(pred[-frames_to_predict:].shape, y_expected[-frames_to_predict:].shape)
                # print(pred[-frames_to_predict:, 0], y_expected[-frames_to_predict:, 0])

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
    parser.add_argument('--save_best', type=bool, default=False) # only save best model
    parser.add_argument('--folder', type=str, required=True) # dataset location
    parser.add_argument('--name', type=str, required=True) # name of the model

    args = parser.parse_args()

    # torch.multiprocessing.set_start_method('spawn')

    frames_per_clip = 5
    frames_to_predict = 1 # must be <= frames_per_clip
    stride = 1 # number of frames to shift when loading clips
    batch_size = 32
    epoch_ratio = 0.25 # to sample just a portion of the dataset
    epochs = 10
    lr = 0.001
    num_workers = 6

    dim_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout_p = 0.1

    trainer = TrainerObs()

    model = TransformerObs(dim_model=dim_model, num_heads=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout_p=dropout_p)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() # TODO: change this to mse + condition + gradient difference
    if args.dataset == 'roboturk':
        train_dataset = RoboTurkObs(num_frames=5, stride=stride, dir=args.folder, stage='train', shuffle=True)
        train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=int(len(train_dataset) * epoch_ratio))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=num_workers)

        test_dataset = RoboTurkObs(num_frames=5, stride=stride, dir=args.folder, stage='test', shuffle=True)
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