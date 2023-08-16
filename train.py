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

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx

from utils import *
from metrics import *
import pickle
import argparse
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler
from model import *
from icecream import ic


def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)


def train(epoch):
    # metrics = {'train_loss': [], 'val_loss': []}
    global metrics, loader_train
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    # for eth, the value of turn_point is 69
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_train):
        # for eth, the maximum of cnt is 70
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]

        # obs_traj = self.obs_traj[start:end, :],
        # pred_traj_gt = self.pred_traj[start:end, :],
        # obs_traj_rel = self.obs_traj_rel[start:end, :],
        # pred_traj_gt_rel = self.pred_traj_rel[start:end, :],
        # non_linear_ped = self.non_linear_ped[start:end],
        # loss_mask = self.loss_mask[start:end, :],
        # V_obs = self.v_obs[index],
        # A_obs = self.A_obs[index],
        # V_tr = self.v_pred[index],
        # A_tr = self.A_pred[index],
        # vgg_list = self.fet_map[self.fet_list[index]]
        # we have to say that there just four elements which we will use: V_obs, A_obs, V_tr and vgg_list.
        # from above, V_obs, A_obs and vgg_list are used for model(V_obs.permute(0, 3, 1, 2), A_obs.squeeze(), vgg_list)
        # V_pred is produced by model(V_obs.permute(0, 3, 1, 2), A_obs.squeeze(), vgg_list), then,
        # V_tr, V_pred are used for graph_loss(V_pred.permute(0, 2, 3, 1).squeeze(), V_tr).
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr, vgg_list = batch
        obs_traj *= scale
        pred_traj_gt *= scale
        obs_traj_rel *= scale
        pred_traj_gt_rel *= scale
        V_obs *= scale
        V_tr *= scale

        optimizer.zero_grad()
        # Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze(), vgg_list)

        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        # for eth, if (batch_count != 128) && (cnt != 69), then code will run 'if sentence'
        if batch_count % args.batch_size != 0 and cnt != turn_point:  # batch_size=128
            l = graph_loss(V_pred, V_tr)
            # ic(l)
            # ic(torch.isnan(V_pred).any())
            # ic(torch.isnan(V_tr).any())
            # ic(torch.isinf(V_pred).any())
            # ic(torch.isinf(V_tr).any())
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l
        else:
            # for eth, when cnt==69, the 'else sentence' will run
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            # Metrics
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)
    metrics['train_loss'].append(loss_batch / batch_count)


def vald(epoch):
    # 2023.4.11
    global metrics, loader_val, constant_metrics_train, constant_metrics_val

    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_val):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr, vgg_list = batch
        obs_traj *= scale
        pred_traj_gt *= scale
        obs_traj_rel *= scale
        pred_traj_gt_rel *= scale
        V_obs *= scale
        V_tr *= scale

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze(), vgg_list)

        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            # Metrics
            loss_batch += loss.item()
            print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['val_loss'].append(loss_batch / batch_count)

    # add codes to save the best models 2023.4.11
    if metrics['train_loss'][-1] < constant_metrics_train['min_train_loss']:
        constant_metrics_train['min_train_loss'] = metrics['train_loss'][-1]
        constant_metrics_train['min_train_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'train_best.pth')
    # 2023.4.11
    if metrics['val_loss'][-1] < constant_metrics_val['min_val_loss']:
        constant_metrics_val['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics_val['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')  # OK


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model specific parameters
    parser.add_argument('--input_size', type=int, default=5)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--n_ssagcn', type=int, default=2, help='Number of ST-GCNN layers')
    parser.add_argument('--n_txpcnn', type=int, default=7, help='Number of TXPCNN layers')
    parser.add_argument('--kernel_size', type=int, default=3)

    # Data specifc paremeters
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--dataset', default='eth',
                        help='eth,hotel,univ,zara1,zara2')

    # Training specifc parameters
    parser.add_argument('--batch_size', type=int, default=450,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=250,
                        help='number of epochs')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')
    parser.add_argument('--lr', type=float, default=0.01,  # 0.01
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=150,
                        help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=False,
                        help='Use lr rate scheduler')
    parser.add_argument('--scale', type=float, default=3.5,
                        help='data_scale')
    parser.add_argument('--tag', default='ssagcn-eth',
                        help='personal tag for the model ')

    args = parser.parse_args()
    print('*' * 30)
    print("Training initiating....")
    print(args)
    # Data prep
    obs_seq_len = args.obs_seq_len
    pred_seq_len = args.pred_seq_len
    # data_set = './datasets/'+args.dataset+'/'
    # dset_train = TrajectoryDataset(
    #         data_set+'train/',
    #         obs_len=obs_seq_len,
    #         pred_len=pred_seq_len,
    #         skip=1,norm_lap_matr=True)
    # dset_train = torch.load("./data/" + args.dataset + "_test.pt")
    dset_train = torch.load("./data/" + args.dataset + "_train.pt")
    loader_train = DataLoader(
        dset_train,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=True,
        num_workers=1)
    # dset_val = TrajectoryDataset(
    #         data_set+'val/',
    #         obs_len=obs_seq_len,
    #         pred_len=pred_seq_len,
    #         skip=1,norm_lap_matr=True)
    dset_val = torch.load("./data/" + args.dataset + "_val.pt")
    # dset_val = torch.load("./data/" + args.dataset + "_test.pt")
    # dset_val = torch.load("./data/" + args.dataset + "_train.pt")
    loader_val = DataLoader(
        dset_val,
        batch_size=1,
        shuffle=False,
        num_workers=1)
    # Defining the model
    model = SocialSoftAttentionGCN(stgcn_num=args.n_ssagcn, tcn_num=args.n_txpcnn,
                                   output_feat=args.output_size, seq_len=args.obs_seq_len,
                                   kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).cuda()
    scale = args.scale
    # Training settings
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.use_lrschd:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
    checkpoint_dir = './checkpoint/' + args.tag + '/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)
    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)
    # Training
    # 2023.4.11
    metrics = {'train_loss': [], 'val_loss': []}
    constant_metrics_val = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}

    # add min_train_loss and epoch 2023.4.11
    constant_metrics_train = {'min_train_epoch': -1, 'min_train_loss': 9999999999999999}

    print('Training started ...')
    for epoch in range(args.num_epochs):
        train(epoch)
        vald(epoch)
        if args.use_lrschd:
            scheduler.step()

        print('*' * 30)
        print('Epoch:', args.tag, ":", epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])

        # 2023.4.11
        print(constant_metrics_val)
        print(constant_metrics_train)

        print('*' * 30)

        with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp:
            pickle.dump(metrics, fp)

        # add 2023.4.11
        with open(checkpoint_dir + 'constant_metrics_val.pkl', 'wb') as fp:
            pickle.dump(constant_metrics_val, fp)

        # add 2023.4.11
        with open(checkpoint_dir + 'constant_metrics_train.pkl', 'wb') as fp:
            pickle.dump(constant_metrics_train, fp)
