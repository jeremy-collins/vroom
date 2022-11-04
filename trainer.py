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
from utils import Utils

from roboturk_loader import RoboTurk

class Trainer():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)
        model = Transformer()
        self.SOS_token = torch.ones((1, model.dim_model), dtype=torch.float32, device=self.device) * 2
        self.utils = Utils()
        self.utils.init_resnet()
            
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
            X = torch.tensor(X).to(self.device)

            X_emb = []
            for clip in X:
                # encode image
                emb = self.utils.encode_img(clip)
                X_emb.append(emb)

            X_emb = torch.stack(X_emb)
            X_emb = X_emb.squeeze(3)
            X_emb = X_emb.squeeze(3)

            # y = batch['y']
            y = X_emb # because the target needs to be in the same vector space as the input.
                      # we will predict a linear projection of the next embedding (see self.out in transformer.py)

            y = torch.tensor(y).to(self.device)
            
            # y_input = y
            # y_expected = y
            
            # shift the tgt by one so we always predict the next embedding
            y_input = y[:,:-1] # all but last 
            # y_input = y # because we don't have an EOS token
            # y_expected = y[:,1:] # all but first because the prediction is shifted by one
            y_expected = batch['y'][:,1:] # all but first joint values
            y_expected = torch.tensor(y_expected).to(self.device)
            
            y_expected = y_expected.reshape(y_expected.shape[0], y_expected.shape[1], -1)
            y_expected = y_expected.permute(1, 0, 2)
            
            # Get mask to mask out the future frames
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(self.device)
        
            # X shape is (batch_size, src sequence length, input.shape)
            # y_input shape is (batch_size, tgt sequence length, input.shape)
            
            # print('X_emb shape: ', X_emb.shape)
            # print('y_input shape: ', y_input.shape)

            pred = model(X_emb, y_input, tgt_mask)
            
            # Permute pred to have batch size first again
            # pred = pred.permute(1, 2, 0)
            
            # loss = loss_fn(pred[-1], y_expected[-1])
            # loss = loss_fn(pred[-frames_to_predict:], y_expected[-frames_to_predict:])
            
            # emb_to_joints = model_dim -> 8, to compare with ground truth
            loss = loss_fn(pred[-frames_to_predict:], y_expected[-frames_to_predict:])
            # print(pred[-frames_to_predict:].shape, y_expected[-frames_to_predict:].shape)

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
                X = torch.tensor(X).to(self.device)

                X_emb = []
                for clip in X:
                    # encode image
                    emb = self.utils.encode_img(clip)
                    X_emb.append(emb)

                X_emb = torch.stack(X_emb)
                X_emb = X_emb.squeeze(3)
                X_emb = X_emb.squeeze(3)

                # y = batch['y']
                y = X_emb # because the target needs to be in the same vector space as the input.
                        # we will predict a linear projection of the next embedding (see self.out in transformer.py)

                y = torch.tensor(y).to(self.device)
                
                # y_input = y
                # y_expected = y
                
                # shift the tgt by one so we always predict the next embedding
                y_input = y[:,:-1] # all but last 
                # y_input = y # because we don't have an EOS token
                # y_expected = y[:,1:] # all but first because the prediction is shifted by one
                y_expected = batch['y'][:,1:] # all but first joint values
                y_expected = torch.tensor(y_expected).to(self.device)
                
                y_expected = y_expected.reshape(y_expected.shape[0], y_expected.shape[1], -1)
                y_expected = y_expected.permute(1, 0, 2)
                
                # Get mask to mask out the future frames
                sequence_length = y_input.size(1)
                tgt_mask = model.get_tgt_mask(sequence_length).to(self.device)
            
                # X shape is (batch_size, src sequence length, input.shape)
                # y_input shape is (batch_size, tgt sequence length, input.shape)
                
                # print('X_emb shape: ', X_emb.shape)
                # print('y_input shape: ', y_input.shape)

                pred = model(X_emb, y_input, tgt_mask)
                
                # Permute pred to have batch size first again
                # pred = pred.permute(1, 2, 0)
                
                # loss = loss_fn(pred[-1], y_expected[-1])
                # loss = loss_fn(pred[-frames_to_predict:], y_expected[-frames_to_predict:])
                
                # emb_to_joints = model_dim -> 8, to compare with ground truth
                loss = loss_fn(pred[-frames_to_predict:], y_expected[-frames_to_predict:])
                # print(pred[-frames_to_predict:].shape, y_expected[-frames_to_predict:].shape)

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
    frames_to_predict = 5 # must be <= frames_per_clip
    stride = 1 # number of frames to shift when loading clips
    batch_size = 32
    epoch_ratio = 1 # to sample just a portion of the dataset
    epochs = 10
    lr = 0.00001
    num_workers = 8

    dim_model = 256
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout_p = 0.1

    trainer = Trainer()
    
    model = Transformer(num_tokens=0, dim_model=dim_model, num_heads=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout_p=dropout_p)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() # TODO: change this to mse + condition + gradient difference
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