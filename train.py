import random
import numpy
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
from sklearn.cluster import DBSCAN
from kmeans.kmean import exft_kms
from scipy.io import savemat
import itertools
import pandas as pd
from sklearn import metrics
from collections import deque
from random import *

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batchsize', type=int, default=4, help='batch size in training')
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

label_rand = np.load("model_feat_20_6.npy", allow_pickle=True).item()
label_rand = label_rand['fts']
label_rand = torch.Tensor(label_rand).to(args.device)
label_rand = label_rand.unsqueeze(0)

def main(load_mode=True, load_name='pretrained.pth', max_epoch=args.epoch, input_class=40):

    if 1:
        global args
        print('加载模型中....')
        model = UniModel(args=args, n_class=input_class)
        model.to(args.device)

        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.9, 0.999),eps=1e-08,weight_decay=args.wd)

    # load
    if load_mode:
        print('加载预训练模型参数中'+load_name)
        pretrained = torch.load('/home/dh/Fallson/Version0/experiment/checkpoints/'+load_name)
        model.load_state_dict(pretrained)

    list_dataset = np.load('kmeans/ori_list.npy', allow_pickle=True).item()

    trainDataset = GetDataTrain(dataType='all', loadlist=False, list=list_dataset)
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchsize, shuffle=True, num_workers=args.j,
                                              pin_memory=True, drop_last=True)
    validDataset = GetDataTrain(dataType='query')
    validLoader = torch.utils.data.DataLoader(validDataset, batch_size=args.batchsize, shuffle=False, num_workers=args.j,
                                              pin_memory=True, drop_last=True)
    warmDataset = GetDataTrain(dataType='target')
    warmLoader = torch.utils.data.DataLoader(warmDataset, batch_size=16, shuffle=False,num_workers=args.j,
                                              pin_memory=True, drop_last=True)

    print("开始训练")
    if 1:
        warmuppath = []
        alllable = torch.tensor([])
        alltarget = torch.tensor([], dtype=int)
        for idx, input_data in enumerate(tqdm(warmLoader)):
            target = input_data['target_mv'].reshape(-1)
            alltarget = torch.cat((alltarget, target), dim=0)
            target = target.to(args.device)
            data_mv = input_data['data_mv'].to(args.device)
            data_pc = input_data['data_pc'].to(args.device)
            data_mesh1 = input_data['data_mesh1'].to(args.device)
            data_mesh2 = input_data['data_mesh2'].to(args.device)
            data_vox = input_data['data_vox'].to(args.device)
            data_path = input_data['path_all']
            warmuppath.extend(data_path)

            model.eval()
            label_rand1 = label_rand.repeat(data_mv.shape[0], 1, 1)
            with torch.no_grad():
                out, fts,_,_ = model(data_mv, data_pc, data_mesh1, data_mesh2, data_vox, True, label_rand1)
                fts1 = fts.detach().cpu()
                alllable = torch.cat((alllable, fts1), dim=0)

        db = DBSCAN(eps=13, min_samples=13)
        size = alllable.shape[0]
        pseudo_lable = np.zeros([size,2])
        alllable , alltarget = alllable.numpy() , alltarget.numpy()
        out_cluster = db.fit_predict(alllable)
        for m,n in enumerate(out_cluster):
            i = 0
            if n != -1:
                i = 1
            pseudo_lable[m, :] = [n , i]

        savelable(warmuppath, pseudo_lable)

#START TRAINING
    for epoch in range(0, max_epoch):   # train
        print("当前epoch为%d"%epoch)
        cur_lr = adjust_learning_rate(optimizer, epoch, max_epoch, gain=0.5)
        for idx, input_data in enumerate(tqdm(trainLoader)):
            if 1:
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
                data_path = input_data['path_all']

                model.train()
                label_rand1 = label_rand.repeat(data_mv.shape[0], 1, 1)
                out, fts,ft1,ft2 = model(data_mv, data_pc, data_mesh1, data_mesh2, data_vox, True, label_rand1)
                loss1 = ce_loss(out,target,mask_lab)
                loss2 = ss_loss(ft1,ft2,anti)
                loss3 = rl_loss(out,anti,data_path,epoch,max_epoch)
                loss = loss1+loss2+loss3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if (epoch + 1) % max_epoch == 0:
            top_acc_path = save_model(model, args.model_name, epoch, 8.0, top=False)
            print("模型已保存")

        #cluster
        if (epoch + 1) % 2 == 0:
            main_path = []
            main_lable = torch.tensor([])
            main_pred = torch.tensor([],dtype=int)
            with torch.no_grad():
                for idx, input_data in enumerate(tqdm(warmLoader)):
                    data_mv = input_data['data_mv'].to(args.device)
                    data_pc = input_data['data_pc'].to(args.device)
                    data_mesh1 = input_data['data_mesh1'].to(args.device)
                    data_mesh2 = input_data['data_mesh2'].to(args.device)
                    data_vox = input_data['data_vox'].to(args.device)
                    data_path = input_data['path_all']
                    main_path.extend(data_path)
                    model.eval()
                    out, fts = model(data_mv, data_pc, data_mesh1, data_mesh2, data_vox, True, label_rand1)
                    _, pred  = torch.max(F.softmax(torch.mean(out.squeeze(2), dim=0)), 1)
                    fts1 = fts.detach().cpu()
                    main_lable = torch.cat((main_lable, fts1), dim=0)
                    main_pred = torch.cat((main_pred, pred), dim=0)

                    para_eps = 14.5 + 0.1*epoch
                    para_ms = 15 - (epoch//10)
                    dbclu = DBSCAN(eps=para_eps, min_samples=para_ms)
                    main_lable, main_pred = main_lable.numpy(), main_lable.numpy()
                    dbout = db.fit_predict(main_lable)
                    cluout = cluprocess(dbout, pred, path)
                    size = main_lable.shape[0]
                    pseudo_lable = np.zeros([size, 2])
                    for m, n in enumerate(cluout):
                        i = 0
                        if n != -1:
                            i = 1
                        pseudo_lable[m, :] = [n, i]
                    savelable(main_path, pseudo_lable)

def savelable(paths,out):

    for i , path in enumerate(paths):
        fullpath = os.path.join(path,'dbtensor.npy')
        np.save(out[i,:],fullpath)

def greedy(target, epoch,max_epoch):

    ep_min=0.01
    ep_max=1
    ratio = torch.max(ep_min, ep_max - (ep_max - ep_min) * epoch / max_epoch)
    random_target = torch.randint(8, 40, (data_mv.shape[0], 1))
    random_target = torch.LongTensor(random_target).reshape(-1).to(args.device)
    rand1 = torch.rand(out.shape[0])
    maxval = target
    for idx,v in enumerate(rand1):
        if v<ratio and target[idx]==-1:
            maxval[idx]=random_target[idx]
    return maxval

def ce_loss(out,target,mask_lab):

    out = torch.mean(out.squeeze(2), dim=0)
    out = out[mask_lab]
    lable = target[mask_lab]
    lable = F.one_hot(lable, num_classes=40).float().to(args.device)
    preds = F.log_softmax(preds , dim=-1)
    loss = (-lable * preds).mean()

    return loss

class ContrastiveLossELI5(nn.Module):
    def __init__(self, batch_size, temperature=0.5, verbose=False):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")

            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size,)).scatter_(0, torch.tensor([i]), 0.0).to(args.device)
            if self.verbose: print(f"1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            if self.verbose: print("Denominator", denominator)

            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2 * N) * loss

def ss_loss(ft1,ft2):

    n = ft1.size()[0]
    crit = ContrastiveLossELI5(batch_size=n, temperature=0.2, verbose=False)
    loss = crit(ft1, ft2)

    return loss

def rl_loss(out,anti,data_path,epoch,max_epoch):

    lable_read = torch.zeros([8, 2]).to(args.device)
    num = 0
    for i,j in enumerate(anti):
        if j:
            try:
                dbpath = data_path[i]
                loadpath = os.path.join(dbpath, 'dbtensor.npy')
                loadarray = np.load(loadpath)
            except:
                loadarray = np.array([-1, 0])
            lable_read[num,:] = torch.tensor(loadarray)
            num += 1
    target = lable_read[:num+1,0]
    random_target = greedy(target, epoch,max_epoch)
    out = torch.mean(out.squeeze(2), dim=0)
    lable = random_target
    pred = F.softmax(out, dim=1)
    loss = torch.tensor([0.]).to(args.device)
    num = torch.tensor([1.]).to(args.device)
    for i,j in enumerate(random_target):
        if j!=-1:
            loss += ((1-pred[i,lable[i]])**2)/2
    loss = loss/num

    return  loss

def cluprocess(dbout, pred, path):
    n = path.shape[0]
    lastdbarray = np.array([n, 2])
    dbout1 = dbout + 50
    for i1, ji in enumerate(dbout):
        try:
            dbpath = path[i]
            loadpath = os.path.join(dbpath,'dbtensor.npy')
            loadarray = np.load(loadpath)
        except:
            loadarray = np.array([-1, 0])
        lastdbarray[i1, :] = loadarray
    lastdbout = lastdbarray[:, 0]
    maxnum = np.max(dbout) + 1
    for i in range(maxnum):
        lastdb = lastdbout[dbout == i]
        nowpred = pred[dbout == i]
        lastdbcount = np.bincount(lastdb + 8)
        nowpredcount = np.bincount(nowpred)
        lastdbsum = np.sum(lastdbcount)
        noisenum = lastdbcount[7]
        if noisenum / lastdbsum < 0.9:
            nonoiselast = lastdbcount[8:]
            pos1 = np.argmax(nonoiselast)
            for i2, j2 in enumerate(lastdbout):
                if j2 == -1:
                    if dbout1[i2] - 50 == i:
                        lastdbout[i2] = pos1
        else:
            nonoisepred = nowpredbcount[8:]
            pos2 = np.argmax(nonoisepred)
            for i2, j2 in enumerate(lastdbout):
                if j2 == -1:
                    if dbout1[i2] - 50 == i:
                        lastdbout[i2] = pos2
    return lastdbout

def adjust_learning_rate(optimizer, epoch, max_epoch, gain=1):
    """Sets the learning ra to the initial LR decayed by 10 every 200 epochs"""
    up   =  gain*0.00002     #gain * 0.0002
    down =  gain * 0.00000005#gain * 0.0000005
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

def snk_loss(flable, ftarget, fanti):

    a = int(sum(fanti).item())
    snk = SinkhornKnopp(outsize=a, err=1e-1)
    m,n = 0,0
    b = torch.zeros((a,32))
    for i in fanti:
        if i:
            b[m,:] = flable[n,8:]
            m +=1
        n+=1

    c = snk(b)
    x,y = 0,0
    for i in fanti:
        if i :
            ftarget[x]=c[y]
            y += 1
        x += 1
    ftarget = torch.tensor(ftarget,dtype=int)

    targets = (
        F.one_hot(ftarget, num_classes=40)
            .float()
            .to(args.device)
    )


    loss = cross_entropy_loss(flable, targets)

    return loss

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

def save_model(model, model_name, epoch, acc, top=True):
    checkpoints = 'experiment/checkpoints'
    print('Save model epoch:%d, acc:%.3f ... ' % (epoch, acc))
    fs = os.path.join(checkpoints, '%s_epoch_%d_acc_%.4f.pth' % (model_name, epoch, acc))
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

def extract_feat(args, load_name, input_class=8):
    model = UniModel(args=args, n_class=input_class)
    model.eval()

    # device_ids = [int(x) for x in args.gpu.split(',')]
    # torch.backends.cudnn.benchmark = True
    # model.cuda(device_ids[0])
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(args.device)

    path_load = os.path.join('/home/dh/Fallson/Version0/experiment/checkpoints', load_name)
    pretrained = torch.load(path_load)
    model.load_state_dict(pretrained)

    queryDataset = GetDataTrain(dataType='query')
    targetDataset = GetDataTrain(dataType='target')

    queryLoader = torch.utils.data.DataLoader(queryDataset, batch_size=16, shuffle=False,
                                                 num_workers=args.j, pin_memory=True, drop_last=False)
    targetLoader = torch.utils.data.DataLoader(targetDataset, batch_size=16, shuffle=False,
                                                 num_workers=args.j, pin_memory=True, drop_last=False)


    ftss_q = []
    lass_q = []
    ftss_t = []
    lass_t = []
    all_lbls, all_preds = [], []

    for idx, input_data in enumerate(tqdm(queryLoader)):
        target = input_data['target_mv'].reshape(-1)
        # target = target.to(args.device)
        data_mv = input_data['data_mv'].to(args.device)
        data_pc = input_data['data_pc'].to(args.device)
        data_mesh1 = input_data['data_mesh1'].to(args.device)
        data_mesh2 = input_data['data_mesh2'].to(args.device)
        data_vox = input_data['data_vox'].to(args.device)
        label_rand1 = label_rand.repeat(data_mv.shape[0], 1, 1)
        if idx == 0:
            name_q = input_data['name_all']
            path_q = input_data['path_all']
            np.save("kmeans/q_name.npy", name_q)
            np.save("kmeans/q_path.npy", path_q)
            del name_q, path_q

        with torch.no_grad():
            out, fts = model(data_mv, data_pc, data_mesh1, data_mesh2, data_vox, True, label_rand1)
            conf, preds = torch.max(out, 1)

        ftss_q.append(fts.cpu().data)
        lass_q.append(target.cpu().data)
    ftss_q = torch.cat(ftss_q, dim=0).numpy()
    lass_q = torch.cat(lass_q, dim=0).numpy()
    np.save("kmeans/q_fts.npy", ftss_q)
    np.save("kmeans/q_las.npy", lass_q)

    for idx, input_data in enumerate(tqdm(targetLoader)):
        target = input_data['target_mv'].reshape(-1)
        # target = target.to(args.device)
        data_mv = input_data['data_mv'].to(args.device)
        data_pc = input_data['data_pc'].to(args.device)
        data_mesh1 = input_data['data_mesh1'].to(args.device)
        data_mesh2 = input_data['data_mesh2'].to(args.device)
        data_vox = input_data['data_vox'].to(args.device)
        label_rand1 = label_rand.repeat(data_mv.shape[0], 1, 1)
        if idx == 0:
            name_t = input_data['name_all']
            path_t = input_data['path_all']
            np.save("kmeans/t_name.npy", name_t)
            np.save("kmeans/t_path.npy", path_t)
            del name_t, path_t

        with torch.no_grad():
            out, fts = model(data_mv, data_pc, data_mesh1, data_mesh2, data_vox, True, label_rand1)
            out = F.softmax(out/0.1, dim=-1)
            conf, preds = torch.max(out, 1)

        ftss_t.append(fts.cpu().data)
        lass_t.append(target.cpu().data)
        all_preds.append(preds.cpu().data)
        all_lbls.append(target.cpu().data)
    ftss_t = torch.cat(ftss_t, dim=0).numpy()
    lass_t = torch.cat(lass_t, dim=0).numpy()
    np.save("kmeans/t_fts.npy", ftss_t)
    np.save("kmeans/t_las.npy", lass_t)

    map_modal = []
    for idx, start_end in enumerate([(0, 512), (512, 1024), (1024, 1536), (1536, 2048), (2048, 3648), (0, 3648)]):
        dist_mat_modal = scipy.spatial.distance.cdist(ftss_q[:, start_end[0]:start_end[1]], ftss_t[:, start_end[0]:start_end[1]], 'cosine')
        map_modal.append(map_score(dist_mat_modal, lass_q, lass_t))
    res = {
        "map_img": map_modal[0],
        "map_pt": map_modal[1],
        "map_mesh": map_modal[2],
        "map_vox": map_modal[3],
        "map_fuse": map_modal[4],
        "map_all": map_modal[5]
    }
    print(res)
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)

    return 0

def Validate(args, model, validateLoader):
    # 各类准确率
    # acc_avg = AverageMeter()
    # acc1_avg = AverageMeter()
    # acc2_avg = AverageMeter()
    # loss_avg = AverageMeter()
    all_lbls, all_preds = [], []
    # fts_img, fts_mesh, fts_pt, fts_vox = [], [], [], []
    all_fts = []

    label_rand = np.load("model_feat_20_6.npy", allow_pickle=True).item()
    label_rand = label_rand['fts']
    label_rand = torch.Tensor(label_rand).to(args.device)
    label_rand = label_rand.unsqueeze(0)

    for idx, input_data in enumerate(tqdm(validateLoader)):
        target = input_data['target_mv'].reshape(-1)
        # target = target.to(args.device)
        data_mv = input_data['data_mv'].to(args.device)
        data_pc = input_data['data_pc'].to(args.device)
        data_mesh1 = input_data['data_mesh1'].to(args.device)
        data_mesh2 = input_data['data_mesh2'].to(args.device)
        data_vox = input_data['data_vox'].to(args.device)
        label_rand1 = label_rand.repeat(data_mv.shape[0], 1, 1)

        with torch.no_grad():
            out, fts = model(data_mv, data_pc, data_mesh1, data_mesh2, data_vox, True, label_rand1)
            _, preds = torch.max(out, 1)

        out = out.cpu().data
        # acc = get_acc_topk(out, target, (1,))
        # acc_avg.update(np.array([acc]).reshape(-1))
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(target.squeeze().detach().cpu().numpy().tolist())
        all_fts.append(fts.detach().cpu().numpy())
    all_fts = np.concatenate(all_fts, axis=0)
    fts_uni = all_fts
    dist_mat = scipy.spatial.distance.cdist(fts_uni, fts_uni, "cosine")
    map_s = map_score(dist_mat, all_lbls, all_lbls)
    acc_mi = acc_score(all_lbls, all_preds, average="micro")
    acc_ma = acc_score(all_lbls, all_preds, average="macro")
    res = {
        "overall acc": acc_mi,
        "meanclass acc": acc_ma,
        "map": map_s
    }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Epoch Done!\n")
    return map_s, res, acc_mi, acc_ma

if __name__ == '__main__':
    main(load_mode=True, load_name='pretrain.pth', max_epoch=args.epoch, input_class=40)
    # extract_feat(args=args, load_name='all40_epoch_29_acc_0.6000.pth', input_class=40)
    # main(load_mode=False, load_name='map47_acc_0.9874_crop7.pth', max_epoch=20, input_class=40)
    # extract_feat(args=args, load_name='all40_epoch_28_acc_0.5423.pth', input_class=40)







