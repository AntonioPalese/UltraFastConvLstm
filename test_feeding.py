from data.dataloader import LaneClsDataset
from utils.utilities import merge_config, get_cfg
import data.custom_transforms as customTransforms
import torchvision.transforms as t
from torch.utils.data import DataLoader
import os


def draw(imgs, paths):
    import matplotlib.pyplot as plt
    import time
    from data.custom_transforms import DeNormalize
    import numpy as np

    for b in range(imgs.shape[0]):
        for l in range(imgs.shape[1]):
            img = (imgs[b,l]).numpy()
            path = paths[l][b].split('CULane')[1][1:]
            folder = os.path.join('feeding_test',*(path.split(os.sep)[:2]))
            print('tosave',os.path.join('feeding_test', path))
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.imsave(os.path.join('feeding_test', path[:-3]+f'_l_{l}_b_{b}.jpg'),(np.clip(img.transpose(1, 2, 0), 0, 1)*255).astype(np.uint8))
            




if __name__ == '__main__':
    cfg = get_cfg()
    cfg = merge_config(cfg)


    size = tuple(cfg.model.size)

    simu_transform = customTransforms.Compose2([
        customTransforms.RandomRotate(6),
        customTransforms.RandomUDoffsetLABEL(100),
        customTransforms.RandomLROffsetLABEL(200)
    ])
    img_transform = t.Compose([
        t.Resize(size),
        t.ToTensor(),
    ])

    dataset = LaneClsDataset(cfg,img_transform=img_transform,
                             simu_transform=simu_transform,
                             target_transform=None,
                             mode='train')
    
    
    loader = DataLoader(dataset=dataset,batch_size=24, 
                      shuffle=True, num_workers=4)

    for b_idx,data_label in enumerate(loader):
        imgs=data_label['imgs']
        cls_labels= data_label['labels']
        paths = data_label['paths']
        
        draw(imgs, paths)

        