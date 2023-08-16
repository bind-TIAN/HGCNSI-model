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


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def ade(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)  # Swap x and y axes in order
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)  # Swap x and y axes in order

        # the x and y axes changed, so now the pred shape is: [2, predicted_length]
        # N represents the npeds
        # T represents the predicted length
        N = pred.shape[0]
        T = pred.shape[1]

        sum_ = 0
        for i in range(N):  # represents the number of ped IDs.
            for t in range(T):  # represents the predicted length.
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N * T)

    return sum_all / All


def collision_rate(pred_traj, pred_traj_gt):
    pred_traj_gt = torch.from_numpy(pred_traj_gt).cuda()
    pred_traj = pred_traj.squeeze(dim=0).permute(2, 0, 1)

    pred_traj = torch.cat([pred_traj, pred_traj_gt], dim=0)
    seq_len, peds, _ = pred_traj.size()
    collision = torch.zeros(seq_len, peds, peds).cuda()
    for s in range(seq_len):
        for h in range(peds):
            collision[s, h, h] = 1
            for k in range(h + 1, peds):
                l2_norm = get_distance(pred_traj[s, h, :], pred_traj[s, k, :])  # 
                collision[s, h, k] = l2_norm
                collision[s, k, h] = l2_norm
    total = seq_len * peds * (peds - 1)
    collision_sum = torch.where(collision < 0.1)[0]
    loss = len(collision_sum) / total
    return torch.tensor(loss)


def fde(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T - 1, T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N)

    return sum_all / All


def seq_to_nodes(seq_):
    # obs_traj=seq_=[1, npeds, 2, observe_length]
    max_nodes = seq_.shape[1]
    # seq_=[npeds, 2, observe_length]
    seq_ = seq_.squeeze()
    # seq_len=observe_length
    seq_len = seq_.shape[2]

    # V:[observe_length, npeds, 2]
    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):  # s represents seq_length
        # step_.shape:[npeds, 2]
        step_ = seq_[:, :, s]
        # h=npeds
        for h in range(len(step_)):  # h represents npeds
            V[s, h, :] = step_[h]

    return V.squeeze()


def nodes_rel_to_nodes_abs(nodes, init_node):
    # nodes:[observe_length, npeds, 2]
    # nodes: torch.Size([8, 2, 2])
    # init_node=V_x[0, :, :].shape: [npeds, 2]
    # init_node=V_x[0, :, :].shape: [2, 2]
    # List with the same output shape but zero result
    # nodes_.shape:[observe_length, npeds, 2]
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):  # observe_length
        for ped in range(nodes.shape[1]):  # npeds
            nodes_[s, ped, :] = np.sum(nodes[:s + 1, ped, :], axis=0) + init_node[ped, :]

    return nodes_.squeeze()


def closer_to_zero(current, new_v):
    dec = min([(abs(current), current), (abs(new_v), new_v)])[1]
    if dec != current:
        return True
    else:
        return False


def bivariate_loss(V_pred, V_trgt):
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr

    sxsy = sx * sy


    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)

    return result
