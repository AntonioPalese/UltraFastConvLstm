import torch.utils.data
import torchvision.transforms as t
from torch.utils.data import DataLoader
from data.dataset import LaneClsDataset, LaneTestDataset
import data.custom_transforms as customTransforms
import os


def generate_loader(cfg, mode):
    size = tuple(cfg.model.size)

    target_transform = t.Compose([
        t.CenterCrop(size),
        #t.ToTensor()
        #customTransforms.FreeScaleMask(size) ,
        #customTransforms.MaskToTensor() ,
    ])
    simu_transform = customTransforms.Compose2([
        customTransforms.RandomRotate(6),
        customTransforms.RandomUDoffsetLABEL(100),
        customTransforms.RandomLROffsetLABEL(200)
    ])
    img_transform = t.Compose([
        t.Resize(size),
        t.ToTensor(),
        t.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
    ])

    dataset = LaneClsDataset(cfg,img_transform=img_transform,
                             simu_transform=simu_transform,
                             target_transform=None,
                             mode=mode)

    #sampler = torch.utils.data.RandomSampler(dataset)
    return DataLoader(dataset=dataset,batch_size=cfg.train.batch_size, 
                      shuffle=True, num_workers=4)

def generate_test_loader(batch_size, data_root, seq_len, mode):
    img_transforms = t.Compose([
        t.Resize((288, 800)),
        t.ToTensor(),
        t.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if mode == 'test':
        test_dataset = LaneTestDataset(data_root,os.path.join(data_root, 'list/test.txt'), seq_len = seq_len, img_transform = img_transforms)
    elif mode == 'validation':
        test_dataset = LaneTestDataset(data_root,os.path.join(data_root, 'list/val.txt'), seq_len = seq_len, img_transform = img_transforms)
    cls_num_per_lane = 18

    
    #sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False)
    return loader

# def generate_test_loader(cfg):
#     size = tuple(cfg.model.size)

#     processor = t.Compose([
#         t.Resize(size),
#         t.ToTensor(),
#         t.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
#     ])

#     dataset = LaneTestDataset(cfg,img_transform=processor)
#     #dataset = LaneTestDataset_V2(cfg,img_transform=processor)

#     return DataLoader(dataset=dataset , batch_size=cfg.test.batch_size , shuffle=False, num_workers=1)

# def generate_multitest_loader(cfg):
#     size = tuple(cfg.model.size)

#     processor = t.Compose([
#         t.Resize(size),
#         t.ToTensor(),
#         t.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
#     ])

#     dataset = LaneTestEnvironmentDataset(cfg,img_transform=processor)

#     return DataLoader(dataset=dataset , batch_size=cfg.test.batch_size , shuffle=False)