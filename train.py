from data.dataloader import generate_loader
from utils.factory import get_optimizer, get_loss_dict
from utils.scheduler import get_scheduler
from utils.utilities import get_cfg,count_parameters, store_cfg
from model.network import parsingNet
from evaluation.eval_wrapper import eval_lane
from tqdm import tqdm
import torch,os,datetime
import numpy as np
from utils.utilities import merge_config
from torch.utils.tensorboard import SummaryWriter
import shutil

def draw(imgs, pred, griding_num):
    import matplotlib.pyplot as plt
    import time
    from data.custom_transforms import DeNormalize
    W = 640
    col_sample = np.linspace(0, W-1, griding_num)
    col_sample_w = col_sample[1]-col_sample[0]
    culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]


    den = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 

    imgs = den(imgs)

    for j in range(imgs.shape[0]):
        for t in range(imgs.shape[1]):
            for l in range(pred.shape[3]):
                for w in range(pred.shape[2]):
                    img = imgs[j, t].numpy()
                    # print(pred.shape)
                    if pred[j, t, w, l]!=griding_num:
                        x = int(pred[j, t, w, l]*col_sample_w)
                        y = culane_row_anchor[w]
                        #print(x, y)
                        img[:, y-3:y+3, x-3:x+3] = 0
            plt.imsave('outputs/img.jpg',(np.clip(img.transpose(1, 2, 0), 0, 1)*255).astype(np.uint8))


def save_model(cfg, net, optimizer, epoch,save_path):
    state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict()}

    dir_name = f'train__lr_{cfg.train.learning_rate}_num_layers_{cfg.model.num_layers}' + cfg.save.note

    save_path = os.path.join(save_path, dir_name)

    
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    assert os.path.exists(save_path)
    model_path=os.path.join(save_path,'ep%03d.pth' % epoch)

    if os.path.exists(model_path):
        assert cfg.save.overwrite == True,"tring to overwrite an existent weight"

    torch.save(state,model_path)

    with open(os.path.join(save_path,'cfg.txt'), 'w') as file:
        file.write(str(cfg))


def inference(net,data_label, device):
    imgs=data_label['imgs']
    cls_labels= data_label['labels']
    paths = data_label['paths']

    imgs,cls_labels=imgs.cuda(device),cls_labels.long().cuda(device)
    cls_out=net(imgs)
    return {'cls_out': cls_out,'cls_labels': cls_labels[:, -1, :, :]}


# def resolve_val_data(results,use_aux):
#     results['cls_out']=torch.argmax(results['cls_out'],dim=1)
#     if use_aux:
#         results['seg_out']=torch.argmax(results['seg_out'],dim=1)
#     return results


def calc_loss(loss_dict,results):
    loss=0
    for i in range(len(loss_dict['name'])):                  
        data_src=loss_dict['data_src'][i]

        datas=[results[src] for src in data_src]

        loss_cur=loss_dict['op'][i](*datas)

        loss+=loss_cur*loss_dict['weight'][i]
    return loss


def train(net,data_loader,loss_dict,optimizer,scheduler,epoch,device):
    net.train()
    sum_loss = 0

    with tqdm(data_loader) as pbar:
        for b_idx,data_label in enumerate(pbar):
            global_step=epoch * len(data_loader) + b_idx
            results=inference(net,data_label,device)
            loss=calc_loss(loss_dict,results)
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("Sum_loss/train", sum_loss/(b_idx+1), epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step(global_step)
            with torch.no_grad():
                sum_loss+=loss.item()
                pbar.set_postfix(mode='Train...', epoch=epoch, loss=loss.item(), sum_loss=sum_loss/(b_idx+1))

    writer.flush()
    return loss,sum_loss

def validate(net,data_loader,loss_dict,epoch,device):
    net.eval()
    sum_loss = 0

    with tqdm(data_loader) as pbar:
        for b_idx,data_label in enumerate(pbar):
            with torch.no_grad():
                global_step=epoch * len(data_loader) + b_idx
                results=inference(net,data_label,device)
                loss=calc_loss(loss_dict,results)
                writer.add_scalar("Loss/validation", loss.item(), global_step)
                writer.add_scalar("Sum_loss/validation", sum_loss/(b_idx+1), epoch)
                sum_loss+=loss.item()
                pbar.set_postfix(mode='Validate...', epoch=epoch, loss=loss.item(), sum_loss=sum_loss/(b_idx+1))
    writer.flush()
    return loss,sum_loss




if __name__ == "__main__":
    cfg = get_cfg()
    cfg = merge_config(cfg)
    net=parsingNet(cfg).cuda(cfg.model.device)
    count_parameters(net)

    writer = SummaryWriter(log_dir='runs/train_lr_' + str(cfg.train.learning_rate) + "_nl_" + str(cfg.model.num_layers))

    if cfg.save.weights_path == "":
        procedure = 'train'
    elif cfg.save.weights_path != "":
        procedure = 'F1-validation'

    print(store_cfg(cfg))

           
    if procedure == 'train':
        train_loader = generate_loader(cfg, mode='train')    
        val_loader = generate_loader(cfg, mode='validation')        

        optimizer=get_optimizer(net, cfg)

        scheduler=get_scheduler(optimizer,cfg,len(train_loader))

        loss_dict=get_loss_dict(cfg=cfg)

        resume=0
        if cfg.save.keep_weights != "":
            resume = cfg.save.resume_epoch
            state_dict_model=torch.load(os.path.join(cfg.save.keep_weights, 'ep%03d.pth' % resume), map_location='cpu')['model']
            state_dict_optim=torch.load(os.path.join(cfg.save.keep_weights, 'ep%03d.pth' % resume), map_location='cpu')['optimizer']
            net.load_state_dict(state_dict_model)
            optimizer.load_state_dict(state_dict_optim)
            resume+=1
        #net=torch.nn.DataParallel(net)
        for epoch in range(resume,cfg.train.epoch):
            loss,sum_loss = train(net,train_loader,loss_dict,optimizer,scheduler,epoch,cfg.model.device)
            with torch.no_grad():
                save_model(cfg,net,optimizer,epoch,save_path='/work/tesi_apalese/checkpoints')
                val_loss,sum_val_loss = validate(net,val_loader,loss_dict,epoch,cfg.model.device)
    elif procedure == 'F1-validation':
        for epoch in range(cfg.save.resume_epoch,cfg.train.epoch):
            state_dict = torch.load(os.path.join(cfg.save.weights_path, 'ep%03d.pth' % epoch), map_location='cpu')
            state_dict_model=state_dict['model']

            net.load_state_dict(state_dict_model)

            score = eval_lane(net, cfg.train.path, cfg.validate.validate_work_dir, cfg.model.griding_num, cfg.model.seq_len, mode='validation', device=cfg.model.device)  
            writer.add_scalar("F1/validation", score, epoch)
            writer.flush()
            
            try:
                shutil.rmtree(cfg.validate.validate_work_dir)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

    writer.close()