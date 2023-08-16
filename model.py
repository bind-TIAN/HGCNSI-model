import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim
from icecream import ic


class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,  # 5
                 out_channels,  # 5
                 kernel_size,  # 8
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size  # 8
        self.conv = nn.Conv2d(
            in_channels,  # 5
            out_channels,  # 5
            kernel_size=(t_kernel_size, 1),  # (1,1)
            padding=(t_padding, 0),  # (0,0)
            stride=(t_stride, 1),  # (1,1)
            dilation=(t_dilation, 1),  # (1,1)
            bias=bias)  # True
        self.pattn = SequentialSceneAttention()
        self.embedding = nn.Linear(10, 5)

    def forward(self, x, A, sequential_scene_attention):
        # x = torch.Size([1, 2, 8, npeds])
        # A = torch.Size([8, npeds, npeds])
        # A:[8, npeds, npeds]
        # sequential_scene_attention.shape:torch.size([1, 8, npeds, 8])
        assert A.size(0) == self.kernel_size  # 8, which represents the observed length.
        T = x.size(2)  # 8, which represents the observed length

        # x=torch.Size([1, 8, npeds, 2])
        x = x.permute(0, 2, 3, 1)

        # x.shape:torch.Size([1, 8, npeds, 2])
        # sequential_scene_attention.shape:torch.size([1, 8, npeds, 8])
        # x.shape:torch.Size([1, 8, npeds, 10])
        x = torch.cat((x, sequential_scene_attention), 3)

        # x.view(-1, 10).shape:torch.size([1*8*npeds, 10])
        # unified_graph.shape:torch.size([1*8*npeds, 5])
        unified_graph = self.embedding(x.view(-1, 10))

        # unified_graph.shape:torch.size([1,8,npeds,5])
        unified_graph = unified_graph.view(1, T, A.size(2), -1)

        # unified_graph.shape:torch.size([1,5,8,npeds])
        unified_graph = unified_graph.permute(0, 3, 1, 2)

        # unified_graph.shape:torch.size([1,5,8,npeds])
        unified_graph = self.conv(unified_graph)

        # unified_graph.shape:torch.size([1,5,8,npeds])
        # unified_graph.shape:torch.size([n,c,t,v])
        # A.shape:torch.Size([8, npeds, npeds])
        # A.shape:torch.Size([t, v, w])
        # gcn_output_features.shape:torch.size([1,5,8,npeds])
        # gcn_output_features.shape:torch.size([n,c,t,w])
        gcn_output_features = torch.einsum('nctv,tvw->nctw', (unified_graph, A))

        # gcn_output_features.contiguous().shape:torch.size([1,5,8,npeds])
        # gcn_output_features.contiguous().shape:torch.size([n,c,t,w])
        # A.shape:torch.size([8,npeds,npeds])
        # A.shape:torch.size([t, v, w])
        return gcn_output_features.contiguous(), A


class SceneAttentionShare(nn.Module):
    def __init__(self,
                 in_channels,  # 2
                 out_channels,  # 5
                 kernel_size,  # (3,8)
                 use_mdn=False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(SceneAttentionShare, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1  # these two conditions should be satisfied, then next code will be run!
        padding = ((kernel_size[0] - 1) // 2, 0)  # rounding down
        self.use_mdn = use_mdn
        gcn_in_channels = 5
        self.gcn = ConvTemporalGraphical(gcn_in_channels, out_channels, kernel_size[1])  # (5,5,8)

        # scene_att.shape:torch.size([npeds,8])
        self.scene_att = SequentialSceneAttention()
        self.embedding = nn.Linear(10, 5)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,  # 5
                out_channels,  # 5
                (kernel_size[0], 1),  # high kernel_size[0] width 1 convolution core:(3,1)
                (stride, 1),  # the convolutional step:(1,1)
                padding,  # (1,0)
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )  # the design of TCN module
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            # self.residual = lambda x: x
            self.residual = nn.Sequential(
                nn.Conv2d(
                    2,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
                # nn.Conv2d(
                #     out_channels,
                #     out_channels,
                #     kernel_size=1,
                #     stride=(stride,1)),
                # nn.BatchNorm2d(out_channels),
            )
        else:
            # code will execute here!
            # self.residual=(2,5,1,(1,1),...)
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A, vgg):
        # x=v,A=a,vgg=vgg
        # for eth_test.pt
        # v/x = torch.Size([1, 2, 8, 2])
        # v/x:[1, 2, observe_length, npeds]
        # a=torch.Size([8, 2, 2])
        # vgg=torch.Size([1, 14, 14, 512])

        # coordinates.shape:torch.size([1, 2, 2])
        # coordinates:[1, 2, npeds]
        # 2023.4.11, here they just use the last x-y coordinates, so this we called STGCNN-w/o-seq.
        # coordinates = x[:, :, -1, :]  # just fetch the last one of coordinates in observed_length time interval.
        # 2023.4.11, here, we revised to change stgcnn-w/o-seq to stgcnn-seq
        coordinates = x[:, :, 0:x.shape[2], :]

        # 8, which represents the observed time length.
        T = x.size(2)

        # coordinates.shape:torch.size([1, 2, 2]),which represents torch.size([1, npeds, 2])
        # coordinates:[1, npeds, 2]
        # 2023.4.11, the coordinates new shape is [1,npeds,T,2]
        coordinates = coordinates.permute(0, 3, 2, 1)
        # coordinates = coordinates.permute(0, 2, 1)

        # sequential_scene_attention.shape:torch.size([npeds,8])
        # eq4 of paper
        sequential_scene_attention = self.scene_att(vgg, coordinates)

        # sequential_scene_attention = sequential_scene_attention.view(-1, T, sequential_scene_attention.shape[1])
        # sequential_scene_attention = sequential_scene_attention.permute(1, 0, 2)
        # sequential_scene_attention = sequential_scene_attention.unsqueeze(0)

        # 2-dimensional convolution of x
        # x = torch.Size([1, 2, 8, 2])
        # x:[1, 2, observe_length, npeds]
        # res.shape:torch.size([1, 5, 8, npeds])
        # res:[1, 5, observe_length, npeds]
        res = self.residual(x)
        # x = torch.Size([1, 2, 8, npeds])
        # A = torch.Size([8, npeds, npeds])
        # A:[8, npeds, npeds]
        # sequential_scene_attention.shape:torch.size([1, 8, npeds, 8])
        # gcn_output_features.shape:torch.size([1,5,8,npeds])
        # gcn_output_features.shape:torch.size([n,c,t,w])
        # A.shape:torch.size([8,npeds,npeds])
        # A.shape:torch.size([t,v,w])
        gcn_output_features, A = self.gcn(x, A, sequential_scene_attention)
        # gcn_output_features2, A = self.gcn(x, A, sequential_scene_attention)
        # gcn_output_features = gcn_output_features1 + gcn_output_features2
        # gcn_output_features.shape:torch.size([1,5,8,npeds])
        # self.tcn(gcn_output_features).shape:torch.size([1,5,8,npeds])
        # res.shape:torch.Size([1,5,8,npeds])
        # gcn_output_features.shape:torch.size([1,5,8,npeds])

        gcn_output_features = self.tcn(gcn_output_features) + res

        if not self.use_mdn:  # will run this branch
            # gcn_output_features.shape:torch.size([1,5,8,npeds])
            gcn_output_features = self.prelu(gcn_output_features)  # code will get here!

        # gcn_output_features.shape:torch.size([1,5,8,npeds])
        # A.shape:torch.size([8,npeds,npeds])
        return gcn_output_features, A


def make_mlp(dim_list):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):  # for example:(dim_in,dim_out)=(1,2),(2,3),(3,4)...
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.PReLU())
    return nn.Sequential(*layers)  # Put each module in their order into nn.Sequential


class SequentialSceneAttention(nn.Module):
    def __init__(self, attn_L=196, attn_D=512, ATTN_D_DOWN=16, bottleneck_dim=8, embedding_dim=10):
        super(SequentialSceneAttention, self).__init__()

        self.L = attn_L  # 196
        self.D = attn_D  # 512
        self.D_down = ATTN_D_DOWN  # 16
        self.bottleneck_dim = bottleneck_dim  # 8
        self.embedding_dim = embedding_dim  # 10

        self.channel_linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        self.spatial_linear = nn.Linear(embedding_dim, embedding_dim)
        self.after_channel_shuffle_linear = nn.Linear(self.D_down + self.embedding_dim, self.bottleneck_dim)
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)  # (2,10)
        self.pre_att_proj = nn.Linear(self.D, self.D_down)  # (512,16)

        mlp_pre_dim = self.embedding_dim + self.D_down  # 10+16=26
        mlp_pre_attn_dims = [mlp_pre_dim, 512, self.bottleneck_dim]  # [26,512,8]
        self.mlp_pre_attn = make_mlp(mlp_pre_attn_dims)  # Sequential(in_fe=26,out_fe=512...;in_fe=512,out_fe=8...)

        self.attn = nn.Linear(self.L * self.bottleneck_dim, self.L)  # (1568,196)

    def forward(self, vgg, end_pos):  # the end_pos means the last observed time step t.
        # the shape of vgg is:[1,14,14,512]
        npeds = end_pos.size(1)  # npeds represents the number of peds
        end_pos = end_pos[0, :, :]  # end_pos.shape: torch.size([npeds,2]),npeds represents the number of persons.

        # 找到时间长度的计算方法
        T_length = end_pos.shape[1]
        curr_rel_embedding = self.spatial_embedding(end_pos)  # curr_rel_embedding.shape:[npeds,10]

        # 把x和y坐标加工成具备输入条件的数据集
        curr_rel_embedding = curr_rel_embedding.view(curr_rel_embedding.shape[0], curr_rel_embedding.shape[1], 1,
                                                     self.embedding_dim).repeat(1, 1, self.L, 1)

        # 由于相机的位置是固定的，因此仅仅需要一帧的信息然后不断的重复就可以作为feature map的特征输入了
        # vgg.shape:torch.size([1,14,14,512])
        vgg = vgg.repeat(end_pos.shape[1] * npeds, 1, 1, 1)
        vgg = vgg.view(-1, self.D)
        features_proj = self.pre_att_proj(vgg)
        features_proj = features_proj.view(-1, self.L, self.D_down)
        features_proj = features_proj.view(-1, T_length, self.L, self.D_down)

        # channel attention,输出形状：[npeds, T_length, self.L, self.D_down]
        xn = nn.functional.adaptive_avg_pool2d(features_proj, (1, 1))
        xn = self.channel_linear(xn)
        xn = features_proj * self.sigmoid(xn)

        # spatial attention,输出形状为：[npeds, T_length, self.L, self.D_down]
        num_channels = T_length
        group_norm = nn.GroupNorm(1, num_channels).cuda()  # 这里的参数1，可以更换成4或8等等。
        xs = group_norm(curr_rel_embedding)
        xs = self.spatial_linear(xs)
        xs = curr_rel_embedding * self.sigmoid(xs)

        # 将xn和xs直接concat，得到out的形状：[npeds, T_length, self.L, self.D_down+self.embedding_dim]
        out = torch.cat([xn, xs], dim=3)

        # channel shuffle，得到out的形状：[npeds, T_length, self.L, self.D_down+self.embedding_dim]
        groups = 2
        bs, chnls, h, w = out.data.size()
        if chnls % groups:  # 2代表了groups
            sequential_scene_attention = out
        else:
            chnls_per_group = chnls // groups
            sequential_scene_attention = out.view(bs, groups, chnls_per_group, h, w)
            sequential_scene_attention = torch.transpose(sequential_scene_attention, 1, 2).contiguous()
            sequential_scene_attention = sequential_scene_attention.view(bs, -1, h, w)

        sequential_scene_attention = sequential_scene_attention.sum(axis=2)
        # 从此处往下添加代码
        # 交换维度
        sequential_scene_attention = sequential_scene_attention.permute(0, 2, 1)

        # 使用1x1卷积
        dimentional1_conv = nn.Conv2d(self.D_down + self.embedding_dim, self.bottleneck_dim, kernel_size=1, stride=1)
        dimentional1_conv = dimentional1_conv.cuda()
        sequential_scene_attention = dimentional1_conv(sequential_scene_attention.unsqueeze(-1))
        sequential_scene_attention = sequential_scene_attention.squeeze(-1)

        # 继续交换一下维度
        sequential_scene_attention = sequential_scene_attention.permute(2, 0, 1)

        # 对维度进行扩展
        sequential_scene_attention = sequential_scene_attention.unsqueeze(0)
        return sequential_scene_attention


class SocialSoftAttentionGCN(nn.Module):
    def __init__(self, stgcn_num=1, tcn_num=5, input_feat=2, output_feat=5,
                 seq_len=8, pred_seq_len=12, kernel_size=3):
        super(SocialSoftAttentionGCN, self).__init__()
        self.stgcn_num = stgcn_num
        self.tcn_num = tcn_num

        self.SceneAttentionShares = nn.ModuleList()
        # SceneAttentionShare(2,5,(3,8))
        self.SceneAttentionShares.append(SceneAttentionShare(input_feat, output_feat, (kernel_size, seq_len)))
        for j in range(1, self.stgcn_num):
            self.SceneAttentionShares.append(SceneAttentionShare(output_feat, output_feat, (kernel_size, seq_len)))

        self.tpcnns = nn.ModuleList()

        # seq_len=8
        # pred_seq_len=12
        # nn.Conv2d(8,12,3,padding=1)
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for j in range(1, self.tcn_num):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))

        # nn.Conv2d(12, 12, 3, padding=1)
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(self.tcn_num):
            self.prelus.append(nn.PReLU())

    def forward(self, v, a, vgg):
        # for eth_test.pt
        # v=torch.Size([1, 2, 8, 2])
        # a=torch.Size([8, 2, 2])
        # vgg=torch.Size([1, 14, 14, 512])
        for k in range(self.stgcn_num):  # self.stgcn_num=1
            # put v and a as input value to the 'SceneAttentionShare' module!
            # gcn_output_features.shape:torch.size([1,5,8,npeds])
            gcn_output_features, a = self.SceneAttentionShares[k](v, a, vgg)

        # gcn_output_features.shape:torch.size([1,8,5,npeds])
        gcn_output_features = gcn_output_features.view(gcn_output_features.shape[0], gcn_output_features.shape[2],
                                                       gcn_output_features.shape[1], gcn_output_features.shape[3])

        # self.tpcnns[0](gcn_output_features).shape:torch.size([1,12,5,npeds])
        # gcn_output_features.shape:torch.size([1,12,5,npeds])
        gcn_output_features = self.prelus[0](self.tpcnns[0](gcn_output_features))

        for k in range(1, self.tcn_num - 1):
            # tcn_output_features.shape:torch.size([1,12,5,npeds])
            tcn_output_features = self.prelus[k](self.tpcnns[k](gcn_output_features)) + gcn_output_features

        # tcn_output_features.shape:torch.size([1,12,5,npeds])
        tcn_output_features = self.tpcnn_ouput(tcn_output_features)

        # tcn_output_features.shape:torch.size([1,5,12,npeds])
        tcn_output_features = tcn_output_features.view(tcn_output_features.shape[0], tcn_output_features.shape[2],
                                                       tcn_output_features.shape[1], tcn_output_features.shape[3])
        # tcn_output_features.shape:torch.size([1,5,12,npeds])
        # a.shape:torch.Size([8, npeds, npeds])
        return tcn_output_features, a
