import numpy as np
import torch
from F_measure import Fmeasure


def converter(data):
    if isinstance(data,torch.Tensor):
        data = data.cpu().data.numpy().flatten()
    return data.flatten()
def fast_hist(label_pred, label_true,num_classes):
    #pdb.set_trace()
    hist = np.bincount(num_classes * label_true.astype(int) + label_pred, minlength=num_classes ** 2)
    hist = hist.reshape(num_classes, num_classes)
    return hist

class Metric_mIoU():
    def __init__(self,class_num):
        self.class_num = class_num
        self.hist = np.zeros((self.class_num,self.class_num))
    def update(self,predict,target):
        predict,target = converter(predict),converter(target)

        self.hist += fast_hist(predict,target,self.class_num)

    def reset(self):
        self.hist = np.zeros((self.class_num,self.class_num))
    def get_miou(self):
        miou = np.diag(self.hist) / (
                    np.sum(self.hist, axis=1) + np.sum(self.hist, axis=0) -
                    np.diag(self.hist))
        miou = np.nanmean(miou)
        return miou

    def get_acc(self):
        acc = np.diag(self.hist) / self.hist.sum(axis=1)
        acc = np.nanmean(acc)
        return acc
    def get(self):
        return self.get_miou()
class MultiLabelAcc():
    def __init__(self):
        self.cnt = 0
        self.correct = 0
    def reset(self):
        self.cnt = 0
        self.correct = 0
    def update(self,predict,target):
        predict,target = converter(predict),converter(target)
        self.cnt += len(predict)
        self.correct += np.sum(predict==target)
    def get_acc(self):
        return self.correct * 1.0 / self.cnt
    def get(self):
        return self.get_acc()
class AccTopk():
    def __init__(self,background_classes,k):
        self.background_classes = background_classes
        self.k = k
        self.cnt = 0
        self.top5_correct = 0
    def reset(self):
        self.cnt = 0
        self.top5_correct = 0
    def update(self,predict,target):
        predict,target = converter(predict),converter(target)
        self.cnt += len(predict)
        background_idx = (predict == self.background_classes) + (target == self.background_classes)
        self.top5_correct += np.sum(predict[background_idx] == target[background_idx])
        not_background_idx = np.logical_not(background_idx)
        self.top5_correct += np.sum(np.absolute(predict[not_background_idx]-target[not_background_idx])<self.k)
    def get(self):
        return self.top5_correct * 1.0 / self.cnt


def update_metrics(metric_dict, pair_data):
    for i in range(len(metric_dict['name'])):
        metric_op = metric_dict['op'][i]
        data_src = metric_dict['data_src'][i]
        metric_op.update(pair_data[data_src[0]], pair_data[data_src[1]])


def reset_metrics(metric_dict):
    for op in metric_dict['op']:
        op.reset()

        



        


                    


        


