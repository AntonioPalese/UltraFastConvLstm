import torch
from PIL import Image
import os
import pdb
import numpy as np

def loader_func(path):
    return Image.open(path)

class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, seq_len, img_transform=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        self.seq_len = seq_len
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane


    def __getitem__(self, index):
        imgs=[]
        names=[]

        for i in range(self.seq_len):
            name = self.list[index*self.seq_len+i].split()[0]
            img_path = os.path.join(self.path, name)
            img = loader_func(img_path)

            if self.img_transform is not None:
                img = self.img_transform(img)

            imgs.append(img)
            names.append(name)

        ret = {'imgs' : torch.stack(imgs, dim=0), 'names' : names}
        return ret

    def __len__(self):
        return len(self.list) // self.seq_len

