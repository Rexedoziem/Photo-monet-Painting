import numpy as np
import os
import random
from config import config
import torchvision.transforms as T
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image

class ImageDataset(data.Dataset):
    def __init__(self, monet_dir, photo_dir, size=(256, 256), normalize=True):
        self.monet_dir = monet_dir
        self.photo_dir = photo_dir
        self.monet_idx = dict()
        self.photo_idx = dict()
        self.transform = transforms.Compose([transforms.Resize(size),T.CenterCrop(size),transforms.ToTensor(),
                                             transforms.Normalize(*config.stats)])
        for i, monet in enumerate(os.listdir(self.monet_dir)):
            self.monet_idx[i] = monet
        for i, photo in enumerate(os.listdir(self.photo_dir)):
            self.photo_idx[i] = photo
        
    def __len__(self):
        return min(len(self.monet_idx.keys()), len(self.photo_idx.keys()))
    
    def __getitem__(self, idx):
        rand_idx = int(np.random.uniform(0, len(self.monet_idx.keys())))
        photo_path = os.path.join(self.photo_dir, self.photo_idx[rand_idx])
        monet_path = os.path.join(self.monet_dir, self.monet_idx[idx])
        photo_img = Image.open(photo_path)
        photo_img = self.transform(photo_img)
        monet_img = Image.open(monet_path)
        monet_img = self.transform(monet_img)
        return photo_img, monet_img