import torch
import torch.nn as nn
import numpy as np
from .image import MVCNN              #没用到
from .mesh import MeshNet
from .voxel import VoxNet
from .pointcloud import PointNetCls    #没用到
from .mdt import TA34
from .fusion import FusionNet
from .pointnet2 import get_model

class UniModel(nn.Module):
    def __init__(self, args, n_class):
        super().__init__()
        self.model_mesh = MeshNet(n_class)
        self.model_pt2 = get_model(n_class)
        # self.model_pt2 = PointNetCls(n_class)
        self.model_vox = VoxNet(n_class)
        self.model_img = TA34(n_class, n_view=20)
        self.model_fuse = FusionNet(n_class, n_view=20)

    def forward(self, img, pt, mesh1, mesh2, vox, global_ft=False, label_rand=None):
        if global_ft:
            # img = img[:, :4, :, :, :]
            out_img, ft_img, gft_img = self.model_img(img)
            out_mesh, ft_mesh, gft_mesh = self.model_mesh([mesh1, mesh2], global_ft)
            out_pt, ft_pt, gft_pt = self.model_pt2(pt, global_ft)
            out_vox, ft_vox = self.model_vox(vox, global_ft)

            ft_ip, out_ip, ftip = self.model_fuse(ft_img, ft_pt, True, label_rand)
            ft_mv, out_mv, ftmv = self.model_fuse(ft_mesh, ft_vox, True, label_rand)
            _, out, fts = self.model_fuse(ft_ip, ft_mv, True, label_rand)
            ft_4 = torch.cat((gft_img, gft_pt, gft_mesh, torch.max(ft_vox, dim=1)[0]), dim=-1)
            ft = torch.cat((ft_4, fts), dim=-1)

            # dec_output = self.classifer(ft)
            # ft = dec_output.reshape(dec_output.shape[0], -1)
            # out = torch.sum(dec_output, dim=1) / (dec_output.shape[1])

            return (out_img[0] + out_pt[0] + out_mesh[0] + out_vox[0] + out[0]) / 5, ft, ftip, ftmv

        else:
            # img = img[:, :4, :, :, :]
            out_img, ft_img, gft_img = self.model_img(img)
            out_mesh, ft_mesh, gft_mesh = self.model_mesh([mesh1, mesh2], global_ft)
            out_pt, ft_pt, gft_pt = self.model_pt2(pt, global_ft)
            out_vox, ft_vox = self.model_vox(vox, global_ft)

            ft_ip, out_ip, _ = self.model_fuse(ft_img, ft_pt, True, label_rand)
            ft_mv, out_mv, _ = self.model_fuse(ft_mesh, ft_vox, True, label_rand)
            _, out, fts = self.model_fuse(ft_ip, ft_mv, True, label_rand)
            out_n = torch.stack((out_img[0],out_pt[0],out_mesh[0],out_vox[0],out_ip[0],out_mv[0],out[0]),dim=0)
            # out_over = torch.stack((out_img[0],out_pt[0],out_mesh[0],out_vox[0],out_ip[0],out_mv[0],out[0]),dim=0)
            ft_4 = torch.cat((gft_img, gft_pt, gft_mesh, torch.max(ft_vox, dim=1)[0]), dim=-1)
            ft = torch.cat((ft_4, fts), dim=-1)
            out_min = torch.min(out_n,dim=2)[0].unsqueeze(-1).repeat(1, 1, 40)
            out_n = (out_n - out_min + 0.01)
            out_max = torch.max(out_n,dim=2)[0].unsqueeze(-1).repeat(1, 1, 40)
            out_n = (out_n / out_max)
            return out_n.unsqueeze(2), ft


class action_net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_l = nn.Linear(3648, 1024)
        self.encoder_f = nn.Linear(3648, 1024)
        self.mlp1 = nn.Sequential(
            nn.Linear(1024 + 1024, 1024),
            nn.Linear(1024, 1024),
            # nn.Linear(1024, 1024),
            nn.Linear(1024, 3648),  # 50mb
        )
    def forward(self, label, ft, target, single=True, fts_opted=None, unlabel_nums = 0, mask=None):
        if single:  # training
            fts_label = torch.zeros(label.shape[0], label.shape[2]).to(self.args.device)
            for i in range(label.shape[0]):
                    fts_label[i] = label[i, target[i], :]
            fts_label = torch.cat((self.encoder_l(fts_label), self.encoder_f(ft)), dim=-1)
            nabla_f = self.mlp1(fts_label)
            ft_next = ft + nabla_f
            loss_act = torch.norm((ft_next - fts_opted), dim=1, p=2) / 60
            return loss_act.sum()/unlabel_nums
        else:
            ft = ft.unsqueeze(1).repeat(1, label.shape[1], 1)
            fts_label = torch.cat((self.encoder_l(label), self.encoder_f(ft)), dim=-1)
            nabla_f = self.mlp1(fts_label.reshape(-1, fts_label.shape[-1]))
            ft_next = ft + nabla_f.reshape(ft.shape[0], ft.shape[1], -1)
            d_met = torch.norm((ft_next - label), dim=-1, p=2)
            return torch.min(d_met,dim=1)[1]    # labeled num: 8


class center_count(nn.Module):
    def __init__(self):
        super().__init__()
        self.nums = torch.zeros(40)
        self.fts = torch.zeros(40, 3648)

    def forward(self, add_fts, add_las):
        for idx, las in enumerate(add_las):
            self.fts[add_las[idx]] = (self.nums[las] * self.fts[las] / (self.nums[las] + 1)) \
                                     + (add_fts[idx] / (self.nums[las] + 1))
            self.nums[las] += 1
        return self.fts

    def finish(self):
        fts_last = self.fts
        print(self.nums)
        self.nums = torch.zeros(40)
        self.fts = torch.zeros(40, 3648)
        print('center_count_done')
        return fts_last#.unsqueeze(0)

    def clear_num(self):
        self.nums = torch.zeros(40)
        self.fts = torch.zeros(40, 3648)
        return 0


if __name__ == "__main__":
    pass
