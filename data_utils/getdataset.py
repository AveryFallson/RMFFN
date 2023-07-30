import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import json
import os
from PIL import Image
import numpy as np
from .binvox_rw import read_as_3d_array

class GetDataTrain(Dataset):
    def __init__(self, dataType='train', imageMode='RGB',views=20, loadlist=False, list=[]):
        self.dt = dataType
        self.imd = imageMode
        self.views = views 

        self.fls = []#/TTarget/bathtub/bathtub_0001/img/bathtub_0001_001
        self.las = []#代表类别的数字
        self.mask_lab = []
        self.names = []#bathtub_0001
        self.paths = []#/TTarget/bathtub/bathtub_0001

        def initloader(path, c_plus=0, mask_val=True):
            clses = sorted(os.listdir(path))   #读Target下文件目录
            for c, cls in enumerate(clses):
                cls_path = os.path.join(path, cls)
                objes = sorted(os.listdir(cls_path))  #读bathtub下文件目录

                for idx, obj_name in enumerate(objes):
                    self.fls_modal = []
                    for modal in ['img', 'mesh', 'pc', 'vox']:
                        modal_path = os.path.join(path, cls, obj_name, modal)
                        modal_objes = sorted(os.listdir(modal_path))
                        views_path = [os.path.join(modal_path, v) for v in modal_objes]
                        self.fls_modal = self.fls_modal + views_path
                    self.fls.append(self.fls_modal)        #获取全部的文件path
                    self.las.append(c + c_plus)            #获取了代表类别的数字
                    self.mask_lab.append(mask_val)
                    # self.names.append(self.fls_modal[0])
                    self.names.append(obj_name)#加入bathtub0001
                    self.paths.append(os.path.join(path, cls, obj_name))
        if loadlist==False:
            if dataType == 'all':
                initloader("/home/dh/Fallson/Data/TTrain/")
                # initloader("/home/dh/Fallson/Data/TTarget/", 8, False)
                initloader("/home/dh/Fallson/Data/TTest/", 8, False)
                # np.save("kmeans/current_list.npy", {'fls': self.fls, 'las': self.las, 'mask_lab': self.mask_lab,
                # 'names': self.names, 'paths': self.paths})

            else:
                if dataType == 'query_target':
                    initloader("/home/dh/Fallson/Data/TTarget/", 8, False)
                    initloader("/home/dh/Fallson/Data/TTest/", 8, False)
                if dataType == 'train':
                    initloader("/home/dh/Fallson/Data/TTrain/")
                if dataType == 'query':
                    initloader("/home/dh/Fallson/Data/TTarget/", 8, True)
                if dataType == 'target':
                    initloader("/home/dh/Fallson/Data/TTest/", 8, False)
        else:
            self.fls = list['fls']
            self.las = list['las']
            self.mask_lab = list['mask_lab']
            self.names = list['names']
            self.paths = list['paths']
            np.save("kmeans/current_list.npy", {'fls': self.fls, 'las': self.las, 'mask_lab': self.mask_lab,
            'names': self.names, 'paths': self.paths})

    def trans(self, path):
        img = Image.open(path).convert('RGB')
        tf = transforms.Compose([
                transforms.RandomRotation(degrees=8), # fill=234),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        img = tf(img)
        return img

    def trans_pc(self, path):
        pt = np.load(path)
        pt = pt - np.expand_dims(np.mean(pt, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(pt ** 2, axis=1)), 0)
        pt = pt / dist  # scale
        pt = torch.from_numpy(pt.astype(np.float32))
        return pt

    def get_mesh(self, mesh_path):
        data = np.load(mesh_path)
        face = data['face']
        neighbor_index = data['neighbor_index']
        max_faces = 1024
        num_point = len(face)
        if num_point < max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(max_faces - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))
        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()
        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)
        return centers, corners, normals, neighbor_index

    def get_vox(self, filename):
        with open(filename, 'rb') as fp:
            vox = read_as_3d_array(fp).data
        vox = torch.from_numpy(vox.astype(np.float32)).unsqueeze(0)
        return vox

    def __getitem__(self, index):
        mask = self.mask_lab[index]
        if mask:
            anti = False
        else:
            anti = True
        fl_dm = self.fls[index][:20]
        target_dm = torch.LongTensor([self.las[index]])
        imgs_dm = []
        for p in fl_dm:
            imgs_dm.append(self.trans(p))
        data_dm = torch.stack(imgs_dm) 

        fl_pc = self.fls[index][21:22]
        target_pc = torch.LongTensor([self.las[index]])
        imgs_pc = []
        for p in fl_pc:
            imgs_pc.append(self.trans_pc(p))
        data_pc = torch.stack(imgs_pc).squeeze(0).transpose(1,0)

        fl_mesh = self.fls[index][20:21]
        target_mesh = torch.LongTensor([self.las[index]])
        centers = []
        corners = []
        normals = []
        neighbor_index = []
        for p in fl_mesh:
            centers1, corners1, normals1, neighbor_index1 = self.get_mesh(p)
            centers.append(centers1)
            corners.append(corners1)
            normals.append(normals1)
            neighbor_index.append(neighbor_index1)
        cen = torch.stack(centers)
        cor = torch.stack(corners)
        nor = torch.stack(normals)
        nei = torch.stack(neighbor_index)
        data_mesh1 = torch.cat((cen, cor, nor), dim=1).squeeze(0)
        data_mesh2 = nei.squeeze(0)

        fl_vox = self.fls[index][22:23]
        target_vox = torch.LongTensor([self.las[index]])
        imgs_vox = []
        for p in fl_vox:
            imgs_vox.append(self.get_vox(p))
        data_vox = torch.stack(imgs_vox)
        data_vox = data_vox.squeeze(0)

        return {'data_mv': data_dm, 'data_pc': data_pc, 'data_mesh1': data_mesh1, 'data_mesh2': data_mesh2, 'data_vox': data_vox,
                'target_mv': target_dm, 'target_pc': target_pc, 'target_mesh': target_mesh, 'target_vox': target_vox, 'name_mv': self.names[index],
                'name_pc': self.names[index], 'name_mesh': self.names[index], 'name_all':self.names, 'anti':anti ,'mask':mask, 'path_all':self.paths[index]}

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.las)


if __name__ == '__main__':
    a = GetDataTrain(dataType='train', imageMode='RGB')
    for i,d in enumerate(a):
        print('i:%d / %d'%(i,len(a)),d['data'].shape, d['target'], d['name'])



# 测试方法
# 1先输出标签和name查看是否对应
# 2 输出某个视图，查看是否对应
