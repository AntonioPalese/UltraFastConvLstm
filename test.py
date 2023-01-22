from utils.utilities import get_cfg,count_parameters
from model.network import parsingNet
from evaluation.eval_wrapper import eval_lane
from tqdm import tqdm
import torch,os
import numpy as np
from utils.utilities import merge_config
from torch.utils.tensorboard import SummaryWriter
import shutil

if __name__ == "__main__":
    cfg = get_cfg()
    cfg = merge_config(cfg)
    net=parsingNet(cfg).cuda(cfg.model.device)
    count_parameters(net)

    writer = SummaryWriter(log_dir='runs/train_lr_' + str(cfg.train.learning_rate) + "_nl_" + str(cfg.model.num_layers))
           
    
    for epoch in range(cfg.save.resume_epoch,cfg.train.epoch):
        state_dict = torch.load(os.path.join(cfg.save.weights_path, 'ep%03d.pth' % epoch), map_location='cpu')
        state_dict_model=state_dict['model']

        net.load_state_dict(state_dict_model)

        score = eval_lane(net, cfg.train.path, cfg.test.test_work_dir, cfg.model.griding_num, cfg.model.seq_len, mode='test', device=cfg.model.device)  
        writer.add_scalar("F1/test", score, epoch)
        writer.flush()
        
        try:
            shutil.rmtree(cfg.test.test_work_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    writer.close()