from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scanpy as sc
import cv2
import numpy as np
import torch

class sciganDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        # M spots and N genes
        # GSM3036911
        # 429 * 16383
        data = open(opt.pos_path, 'r').readlines()
        self.images = self.reshape_data(data)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        img = torch.from_numpy(img).unsqueeze(0) # 1 * 128 * 128
        z = np.random.normal(0,1, (self.opt.latent_dim))
        z = torch.from_numpy(z)
        return img, z

    def name(self):
        return 'sciganDataset'
    
    def reshape_data(self, data):
        # read header
        line = data[0]
        headers = line.strip().split('\t')
        
        imgs = []
        for line in data[1:]:
            img_list = line.strip().split('\t')
            # first element is the spot
            img_list = img_list[1:]
            img_list_int = [int(i) for i in img_list]
            img = np.array(img_list_int)
            img_pad = np.pad(img, (0, 1)) # 16384 = 128 * 128
            imgs.append(img_pad.reshape(self.opt.img_size, self.opt.img_size).astype('float'))
        return imgs

        
    


