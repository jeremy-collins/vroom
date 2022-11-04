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
from roboturk_loader import RoboTurk

class Utils():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def init_resnet(self, freeze=True):
        self.resnet50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        # remove last layer
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.resnet50.to(self.device)
        # freeze resnet50
        if freeze:
            print('using frozen ResNet!')
            self.resnet50.eval()
            for param in self.resnet50.parameters():
                param.requires_grad = False
        else:
            print('not freezing ResNet!')
            self.resnet50.train()
            
    def encode_img(self, img):
        # input image into CNN
        # img = np.array(img, dtype=np.float32)
        # img = cv2.resize(img, (224, 224))
        latents = self.resnet50(img)
        # img = torch.tensor(latents).to(self.device)
        return latents