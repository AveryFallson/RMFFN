from utils import *
from utils2 import split_trainval, AverageMeter, res2tab, acc_score, map_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import time
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import io
import scipy.spatial
import numpy as np
from data_utils.getdataset import GetDataTrain
from models import UniModel, action_net, center_count
import os
import csv
import codecs
from kmeans.kmean import exft_kms
from scipy.io import savemat
import itertools
import pandas as pd


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batchsize', type=int, default=8, help='batch size in training')
    parser.add_argument('--epoch', default=15, type=int, help='number of epoch in training')
    parser.add_argument('--j', default=4, type=int, help='number of epoch in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training SGD or Adam')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--stage', type=str, default='train', help='train test, extract feature')
    parser.add_argument('--views', default=20, type=int, help='the number of views')
    parser.add_argument('--num_classes', default=40, type=int, help='the number of classes')
    parser.add_argument('--model_name', type=str, default='all40', help='train test')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--c', default=5.0, type=float)
    parser.add_argument('--r', default=4, type=int)
    parser.add_argument('--word_dim', default=512, type=int)
    return parser.parse_args()


args = parse_args()
args.device = torch.device('cuda:%s' % args.gpu)
#--------------------------------------------
# 可以针对全局，也可以针对局部
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, max_epoch, gain=1):
    """Sets the learning ra to the initial LR decayed by 10 every 200 epochs"""
    up   =  gain * 0.0002
    down =  gain * 0.0000005
    lrs = (up - (down)) * (np.cos([np.pi * i / max_epoch for i in range(max_epoch)]) / 2 + 0.5) + (down)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrs[epoch]
        print('Learning Rate: {lr:.6f}'.format(lr=param_group['lr']))
    return lrs[epoch]

def cross_entropy_loss(preds, targets):
    temperature = 0.1
    preds = F.log_softmax(preds / temperature, dim=-1)
    return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)

def swapped_prediction(logits, targets):
        loss = 0
        for view in range(7):
            for other_view in np.delete(range(7), view):
                loss += cross_entropy_loss(logits[other_view], targets[view])
        return loss / (7 * (7 - 1))

def ce_loss(logits, logits_over, labels, mask_lab, anti):
    logits = logits.transpose(2, 1)
    logits_over = logits_over.transpose(2, 1)

    nlc = 8
    outputs_unlab = logits[:,:,:,8:]
    outputs_unlab_over = logits_over[:,:,:,8:]

    targets_lab = (
        F.one_hot(labels, num_classes=40)
            .float()
            .to(args.device)
    )
    targets = torch.zeros_like(logits)
    targets_over = torch.zeros_like(logits_over)

    for v in range(7):
        for h in range(1):
            targets[v, h, :, :] = targets_lab.type_as(targets)
            # targets[v, h, anti, nlc:] = snk(outputs_unlab[v, h, anti]).type_as(targets)

            targets_over[v, h, :, :40] = targets_lab.type_as(targets)
            # targets_over[v, h, anti, nlc:] = snk(outputs_unlab_over[v, h, anti]).type_as(targets)

    # compute swapped prediction loss
    loss_cluster = swapped_prediction(logits, targets)
    loss_overcluster = swapped_prediction(logits_over, targets_over)
    return loss_cluster.mean(), loss_overcluster.mean()

label_rand = np.load("model_feat_20_6.npy", allow_pickle=True).item()
label_rand = label_rand['fts']
label_rand = torch.Tensor(label_rand).to(args.device)
label_rand = label_rand.unsqueeze(0)

def main(load_mode=True, load_name='final.pth', max_epoch=10, input_class=40):
    if 1:
        global args

        top_acc = 0.0
        top_acc_path = ''
        acc_avg = AverageMeter()
        model = UniModel(args=args, n_class=input_class)

        if args.gpu == '0,1':
            device_ids = [int(x) for x in args.gpu.split(',')]
            torch.backends.cudnn.benchmark = True
            model.cuda(device_ids[0])
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        elif args.gpu == '0' or args.gpu == '1':
            model.to(args.device)

        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.wd
            )

    print('获取数据集。。。。。')
    list_dataset = np.load('kmeans/ori_list.npy', allow_pickle=True).item()
    PreDataset = GetDataTrain(dataType='train', loadlist=False, list=list_dataset)
    PreLoader = torch.utils.data.DataLoader(PreDataset, batch_size=args.batchsize, shuffle=True, num_workers=args.j,
                                              pin_memory=True, drop_last=True)
    print('START TRAINNING')
    for epoch in range(0, max_epoch):
        cur_lr = adjust_learning_rate(optimizer, epoch, max_epoch, gain=0.5)
        print("epoch TRAINING,CURRENT:%.3f"%epoch)

        for idx, input_data in enumerate(tqdm(PreLoader)):
            target = input_data['target_mv'].reshape(-1)
            target = target.to(args.device)
            data_mv = input_data['data_mv'].to(args.device)
            data_pc = input_data['data_pc'].to(args.device)
            data_mesh1 = input_data['data_mesh1'].to(args.device)
            data_mesh2 = input_data['data_mesh2'].to(args.device)
            data_vox = input_data['data_vox'].to(args.device)
            mask_lab = input_data['mask'].reshape(-1)
            mask_lab = mask_lab.to(args.device)
            anti = input_data['anti'].reshape(-1)
            anti = anti.to(args.device)

            model.train()
            label_rand1 = label_rand.repeat(data_mv.shape[0], 1, 1)
            out, fts,_,_ = model(data_mv, data_pc, data_mesh1, data_mesh2, data_vox, False, label_rand1)
            loss, _ = ce_loss(out, out, target, mask_lab, anti)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) == max_epoch:
            top_acc_path = save_model(model, args.model_name, epoch, top=False)

def save_model(model, model_name, epoch, top=True):
    checkpoints = 'experiment/checkpoints'
    print('Save model epoch:%d, acc:%.3f ... ' % (epoch,1))
    fs = os.path.join(checkpoints, '%s_epoch_%.4f.pth' % (model_name, epoch))
    torch.save(model.state_dict(), fs)
    if top:
        torch.save(model.state_dict(), 'experiment/checkpoints/%s_top.pth' % model_name)
        print('Save model of top acc ...')
    return fs

def load_model(model, path):
    pretrained = torch.load(path)
    model.load_state_dict(pretrained)

def get_acc_of_out(out, target):
    choice = out.max(1)[1]
    correct = choice.eq(target.long()).sum()
    return correct.item() / float(len(target))

def get_acc_topk(out, target, topk=(1,)):
    batch_size = target.shape[0]
    topkm = max(topk)
    _, pred = out.topk(topkm, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    acc = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        acc.append(correct_k.mul_(1.0 / batch_size))
    return np.array(acc)

if __name__ == '__main__':
    main()
#     # logger_kms = get_logger('kmeans', 'kmeans')
#
#     main(load_mode=False, load_name='map47_acc_0.9874_crop7.pth', max_epoch=20, input_class=40)
#     # extract_feat(args=args, load_name='all40_epoch_28_acc_0.5423.pth', input_class=40)
#     # exft_kms(0, 5, logger_kms)
#
#
#
#



