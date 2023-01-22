from data.dataloader import generate_test_loader
import os, json, torch
import numpy as np
import platform
from tqdm import tqdm
from scipy import special


def generate_lines(out, shape, names, output_path, griding_num, localization_type='abs', flip_updown=False):

    col_sample = np.linspace(0, shape[1] - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    for j in range(out.shape[0]):
        out_j = out[j].data.cpu().numpy()
        if flip_updown:
            out_j = out_j[:, ::-1, :]
        if localization_type == 'abs':
            out_j = np.argmax(out_j, axis=0)
            out_j[out_j == griding_num] = -1
            out_j = out_j + 1
        elif localization_type == 'rel':
            prob = special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == griding_num] = 0
            out_j = loc
        else:
            raise NotImplementedError
        name = names[j]

        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            fp.write(
                                '%d %d ' % (int(out_j[k, i] * col_sample_w * 1640 / 800) - 1, int(590 - k * 20) - 1))
                    fp.write('\n')

def run_test(net, data_root, exp_name, work_dir, griding_num, seq_len, mode='test', batch_size=8,device=0):
    # torch.backends.cudnn.benchmark = True
    output_path = os.path.join(work_dir, exp_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    loader = generate_test_loader(batch_size, data_root,seq_len,mode)
    # import pdb;pdb.set_trace()
    for i, data in enumerate(tqdm(loader)):
        imgs, names = data['imgs'], data['names']
        imgs = imgs.cuda(device)
        with torch.no_grad():
            out = net(imgs)

        generate_lines(out,imgs[0,0,0].shape,names[-1],output_path,griding_num,localization_type = 'rel',flip_updown = True)
   

def eval_lane(net, data_root, work_dir, griding_num, seq_len,mode='test', device=0):
    with torch.no_grad():
        net.eval()    
        run_test(net,data_root, 'culane_eval_tmp', work_dir, griding_num, seq_len, mode=mode,device=device,batch_size=1)     

