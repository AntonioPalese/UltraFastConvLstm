class LaneTestDataset_V2(torch.utils.data.Dataset):
    def __init__(self,cfg,img_transform=None):
        super(LaneTestDataset_V2, self).__init__()
        self.path = cfg.train.path
        self.img_transform = img_transform
        self.griding_num = cfg.model.griding_num
        self.seq_len = cfg.model.seq_len
        self.row_anchors = cfg.model.row_anchors
        self.num_lanes = cfg.model.num_lanes
        list_path = cfg.test.list_path
        self.size = cfg.model.size

        with open(list_path,'r') as f:
            self.list=f.readlines()

    def __getitem__(self, index):

        imgs=[]
        cls_labels=[]
        img_paths=[]
    

        for i in range(self.seq_len):
            l=self.list[index+i]
            l_info=l.split()
            img_name,label_name=l_info[0],l_info[1]
            if img_name[0] == '/':
                img_name=img_name[1:]
                label_name=label_name[1:]

            label_path=os.path.join(self.path,label_name)
            label=loader_func(label_path)


            img_path=os.path.join(self.path,'training_validation',img_name)
            img=loader_func(img_path)


            lane_pts=_get_index(label, self.row_anchors, self.num_lanes, self.size[0])
            # get the coordinates of lanes at row anchors

            w,h=img.size
            cls_label=_grid_pts(lane_pts,self.griding_num,w)

            if self.img_transform is not None:
                img=self.img_transform(img)
            
            imgs.append(img)
            cls_labels.append(torch.from_numpy(cls_label))
            img_paths.append(img_path)

        return {'imgs' : torch.stack(imgs, dim=0), 'labels' : torch.stack(cls_labels, dim=0), 'paths' : img_paths}


    def __len__(self):
        return len(self.list) - self.seq_len
        #return len(self.list)


class LaneClsDataset_V2(torch.utils.data.Dataset):
    def __init__(self,cfg,img_transform=None, 
                          target_transform=None, 
                          simu_transform=None,
                          segment_transform=None):

        super(LaneClsDataset_V2, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform
        self.path = cfg.train.path
        self.griding_num = cfg.model.griding_num
        #self.use_aux = cfg.model.use_aux
        self.num_lanes = cfg.model.num_lanes
        self.seq_len = cfg.model.seq_len
        self.row_anchors = cfg.model.row_anchors
        list_path = cfg.train.list_path
        self.size = cfg.model.size

        with open(list_path, 'r') as f:
            self.list = f.readlines()       


    def __getitem__(self, index):

        imgs = []
        cls_labels = []
        img_paths = []

        for i in range(self.seq_len):
            l = self.list[index+i]
            l_info = l.split(' ')

            img_name, label_name = l_info[0], l_info[1]
            if img_name[0] == '/':
                img_name = img_name[1:]
                label_name = label_name[1:]

            label_path = os.path.join(self.path, label_name)

            label = loader_func(label_path)

            img_path = os.path.join(self.path,'training_validation',img_name)            
            img_paths.append(img_path)

            img = loader_func(img_path)

            if self.simu_transform:
                img, label = self.simu_transform(img, label)

            lane_pts = _get_index(label, self.row_anchors, self.num_lanes, self.size[0])

            # get the coordinates of lanes at row anchors

            w, h = img.size
            cls_label = _grid_pts(lane_pts, self.griding_num, w)
            # make the coordinates to classification label
            '''if self.use_aux:
                assert self.segment_transform is not None
                seg_label = self.segment_transform(label)'''


            if self.img_transform:
                img = self.img_transform(img)

            imgs.append(img)
            cls_labels.append(torch.from_numpy(cls_label))

            '''if self.use_aux:
                seg_labels.append(seg_label)'''

            '''if self.use_aux:
                return torch.stack(imgs, dim=0), torch.stack(cls_labels, dim=0), torch.stack(seg_labels, dim=0),img_paths
            '''    
        return {'imgs' : torch.stack(imgs, dim=0), 'labels' : torch.stack(cls_labels, dim=0), 'paths' : img_paths}

    def __len__(self):
        return len(self.list) - self.seq_len