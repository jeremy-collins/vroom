import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from transformer import Transformer
from roboturk_loader import RoboTurk
import PIL
import cv2
import os
import argparse
from utils import Utils

def predict(model, input_sequence, max_length=5):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
    # y_input = torch.ones((1, model.dim_model), dtype=torch.float32, device=device) * 2 # SOS token
    # SOS_token = torch.ones((1, 1, model.dim_model), dtype=torch.float32, device=device) * -100
    SOS_token = torch.ones((1, 1, 2048), dtype=torch.float32, device=device) * -100
    EOS_token = torch.ones((1, 1, model.dim_model), dtype=torch.float32, device=device) * 3
    # y_input = torch.tensor([SOS_token], dtype=torch.float32, device=device)
    y_input = SOS_token
    
    # num_tokens = len(input_sequence[0])
    with torch.no_grad():
        # for _ in range(6):
        # for _ in range(max_length):
        y_input = torch.cat((SOS_token, input_sequence), dim=1) # TODO: change input_sequence to have dim 256
        y_input = model.embedding(y_input)
        print('new y_input size: ', y_input.size())
        # Get target mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        
        pred = model(input_sequence, y_input, tgt_mask) # (batch_size, seq_len, dim_model)
        
        # Permute pred to have batch size first again
        pred = pred.permute(1, 0, 2)
        # # new shape: (batch_size, seq_len, dim_model)
        
        # # X shape is (batch_size, src sequence length, input.shape)
        # # y_input shape is (batch_size, tgt sequence length, input.shape)
    
        # # next item is the last item in the predicted sequence
        # next_item = pred[:,-1,:].unsqueeze(1)
        # # next_item = torch.tensor([[next_item]], device=device)

        # # Concatenate previous input with prediction
        # y_input = torch.cat((y_input, next_item), dim=1)

        # # Stop if model predicts end of sentence
        # # if next_item.view(-1).item() == EOS_token:
        # #     break

    # return y_input.view(-1).tolist()
    # return pred[0,0].view(-1).tolist()
    return pred[0, -1] # return last item in sequence
    
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, required=True)
    parser.add_argument('--pred_frames', type=int, default=1) # number of frames to predict
    parser.add_argument('--show', type=bool, default=False)
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--name', type=str, default='default')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer()
    model.load_state_dict(torch.load('./checkpoints/model_' + str(args.index) + '.pt'))
    model.eval()
    model = model.to(device)
    
    utils = Utils()
    utils.init_resnet()
    
    test_dataset = RoboTurk(num_frames=5, stride=1, dir=args.folder, stage='test', shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = torch.tensor([], device=device)
            preds = torch.tensor([], device=device)
            X = batch['data']
            y = batch['y']

            X = torch.tensor(X, dtype=torch.float32, device=device)

            # shift the tgt by one so with the <SOS> we predict the token at pos 1
            y = torch.tensor(y[:,:-1], dtype=torch.float32, device=device)

            X_emb = []
            for clip in X:
                # encode image
                emb = utils.encode_img(clip)
                X_emb.append(emb)

            X_emb = torch.stack(X_emb)

            X_emb = X_emb.squeeze(3)
            X_emb = X_emb.squeeze(3)

            pred = predict(model, X_emb)

            print('pred: ', pred)
            print('y: ', y[0, -1])

            # for idx, input in enumerate(X_emb.squeeze(0)): # for each input frame
            #     if idx == 0:
            #             continue # SOS token
            #     else:
            #         inputs = torch.cat((inputs, input.unsqueeze(0).unsqueeze(0)), dim=1)
            #         print('inputs shape: ', inputs.shape)

            # for iteration in range(args.pred_frames):
            #     pred = predict(model, X_emb)
            #     pred = torch.tensor(pred, dtype=torch.float32, device=device)
            #     preds = torch.cat((preds, pred.unsqueeze(0).unsqueeze(0)), dim=1)
            #     print('preds shape: ', pred.shape)
            #     all_latents = torch.cat([inputs[:,:-1], preds], dim=1)
            #     X_emb = all_latents[:, -5:] # the next input is the last 5 frames of the concatenated inputs and preds
            #     print('X after modifying: ', X_emb.shape)

            # if args.show:
            #     for latent in all_latents.squeeze(0):
            #         print('latent:', latent)


    #     # counting number of files in ./checkpoints
    #     folder_index = len(os.listdir('./images'))   
    #     os.mkdir('./images/' + str(folder_index))
    #     img_path = os.path.join('./images', str(folder_index), str(index_list[-1].item()) + '_pred.png')
    #     pred_img[0].save(img_path)
    #     # pred_img[0].show()