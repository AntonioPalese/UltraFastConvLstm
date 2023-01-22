from addict import Dict
import yaml
from prettytable import PrettyTable
from PIL import Image

def get_cfg(path='configurations/configuration.yaml'):
    stream = open(path, 'r')
    configuration_dictionary = yaml.load(stream, Loader=yaml.FullLoader)

    return Dict(configuration_dictionary)


def count_parameters(model):
    # thanks to https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model#:~:text=To%20get%20the%20parameter%20count,name%20and%20the%20parameter%20itself.&text=Show%20activity%20on%20this%20post.,-If%20you%20want
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def store_cfg(cfg : Dict, file : str = None):
    if file:
        yaml.dump(cfg.to_dict(), file)        
    elif not file:
        p_cfg = yaml.dump(cfg.to_dict())
        return p_cfg

def get_saving_path(cfg,base_save_path = '/work/tesi_apalese/checkpoints'):
    dir_name = f'train__lr_{cfg.train.learning_rate}_num_layers_{cfg.model.num_layers}' + cfg.save.note

    savepath = os.path.join(base_save_path, dir_name)

    return savepath

def merge_resume_config(cfg):
    path_to_merge = os.path.join(get_saving_path(cfg), 'cfg.yaml')
    if os.path.exists(path_to_merge):
        cfg_to_merge = get_cfg(path_to_merge)
        cfg.save.best_sum_val_loss = cfg_to_merge.save.best_sum_val_loss
        cfg.save.best_score = cfg_to_merge.save.best_score
        cfg.train.current_epoch = cfg_to_merge.train.current_epoch
    return Dict(cfg)
        


   



############################################################################## command line args #####################################################

import os, argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_lanes', default = None, type = int)
    parser.add_argument('--num_layers', default = None, type = int)
    parser.add_argument('--hidden_sizes', default = None, nargs='+', type = int)
    parser.add_argument('--kernel_sizes', default = None, nargs='+', type = int)
    parser.add_argument('--size', default = None, nargs='+', type = int)
    parser.add_argument('--seq_len', default = None, type = int)
    parser.add_argument('--cls_dim', default = None, nargs='+', type = int)
    parser.add_argument('--dataset', default = None, type = str)
    parser.add_argument('--device', default = None, type = int)
    parser.add_argument('--path', default = None, type = str)
    parser.add_argument('--epoch', default = None, type = int)
    parser.add_argument('--validate_work_dir', default = None, type = str)
    parser.add_argument('--batch_size', default = None, type = int)
    parser.add_argument('--optimizer', default = None, type = str)
    parser.add_argument('--learning_rate', default = None, type = float)
    parser.add_argument('--weight_decay', default = None, type = float)
    parser.add_argument('--momentum', default = None, type = float)
    parser.add_argument('--scheduler', default = None, type = str)
    parser.add_argument('--steps', default = None, type = int, nargs='+')
    parser.add_argument('--gamma', default = None, type = float)
    parser.add_argument('--warmup', default = None, type = str)
    parser.add_argument('--warmup_iters', default = None, type = int)
    parser.add_argument('--backbone', default = None, type = str)
    parser.add_argument('--griding_num', default = None, type = int)
    parser.add_argument('--sim_loss_w', default = None, type = float)
    parser.add_argument('--test_batch_size', default = None, type = int)
    parser.add_argument('--shp_loss_w', default = None, type = float)
    parser.add_argument('--note', default = None, type = str)
    parser.add_argument('--weights_path', default = None, type = str)
    parser.add_argument('--keep_weights', default = None, type = str)
    parser.add_argument('--resume_epoch', default = None, type = int)
    parser.add_argument('--test_work_dir', default = None, type = str)
    parser.add_argument('--best_sum_val_loss', default = None, type = float)
    parser.add_argument('--best_score', default = None, type = float)


    return parser

def merge_config(cfg : Dict):
    args = get_args().parse_args()
    items = ['num_lanes', 'num_layers', 'hidden_sizes','kernel_sizes', 'size','seq_len','cls_dim', 
            'dataset','device','path','epoch','validate_work_dir','batch_size','optimizer','learning_rate',
            'weight_decay','momentum','scheduler','steps','gamma','warmup','warmup_iters','backbone','griding_num','sim_loss_w',
            'test_batch_size','shp_loss_w','note','weights_path', 'keep_weights','resume_epoch','test_work_dir','num_lanes',
            'best_sum_val_loss', 'best_score']

    for item in items:
        if getattr(args, item):
            for k,v in cfg.items():
                if isinstance(v,Dict):
                    if item in v.keys():
                        cfg[k][item] = getattr(args, item)
    return Dict(cfg)