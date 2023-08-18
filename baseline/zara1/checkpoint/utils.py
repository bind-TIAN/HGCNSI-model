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
import random
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time
import pickle
from icecream import ic


def softmax_operation(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix))


def get_distance(p1, p2):  # Get scalar values of distances between ped i and ped j
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_norm(p):  # Get scalar values of speed
    return math.sqrt(p[0] ** 2 + p[1] ** 2)


def compute_adjacent_matrix(test_matrix, weight_matrix, ni_degree_matrix, test_matrix_T):  # add 4.4
    '''
    compute adjacent matrix
    '''
    res = test_matrix.dot(weight_matrix).dot(ni_degree_matrix).dot(test_matrix_T)
    return res


def find_hyperedge_weight(edge_weight_dict, edge_i, edge_j):  # add 4.4
    '''
    find the weights of hyperedges
    '''
    if edge_i > edge_j:
        hyperedge_weight_j_i = edge_weight_dict[(edge_j, edge_i)]
        return hyperedge_weight_j_i
    else:
        hyperedge_weight_i_j = edge_weight_dict[(edge_i, edge_j)]
        return hyperedge_weight_i_j


def compute_weight_matrix(test_matrix, hyperedge_neighbor, hyperedge_weight):  # add 4.4
    '''
    compute weight matrix
    '''
    hyperedges_weights = []
    for column in range(test_matrix.shape[1]):  # traverse every hyperedges
        neighbors_list = hyperedge_neighbor[column]  # find neighbors for each hyperedge
        res_column = 0
        for i in range(len(neighbors_list)):
            res_column = res_column + find_hyperedge_weight(hyperedge_weight, column, neighbors_list[i])
        hyperedges_weights.append(res_column)
    return hyperedges_weights


def compute_hyperedges_weight(test_matrix):  # add 4.4
    '''
    compute the weights of hyperedges
    '''
    hyperedge_weight_dict = {}
    for i in range(test_matrix.shape[1]):
        for j in range(i + 1, test_matrix.shape[1]):
            res = test_matrix[:, i] + test_matrix[:, j]
            key = (i, j)
            common_nodes = [i for i, e in enumerate(res) if e == 2]
            hyperedge_weight_dict[key] = len(common_nodes)
    return hyperedge_weight_dict


def compute_hypernodes_degree(test_matrix):  # add 4.4
    '''
    compute hypernodes' degree
    '''
    hypernodes_degree_list = []
    for row in range(test_matrix.shape[0]):
        res = np.sum(test_matrix[row, :])
        hypernodes_degree_list.append(res)
    return hypernodes_degree_list


def compute_hyperedges_degree(test_matrix):  # add 4.4
    '''
    compute the degree of hyperedges
    '''
    hyperedges_degree_list = []
    for column in range(test_matrix.shape[1]):
        res = np.sum(test_matrix[:, column])
        hyperedges_degree_list.append(res)
    return hyperedges_degree_list


def find_nodes_for_each_edge(list):  # return the index of nodes in the matrix.add 4.4
    '''
    find nodes for each edges
    '''
    list_1_index = []
    for i, ele in enumerate(list):
        if ele == 1:
            list_1_index.append(i)
    return list_1_index


def find_hyperedge_neighbor(test_matrix):  # add 4.4
    '''
    find neighbors of a specific hyperedge
    '''
    edge_neighbor = []
    for column in range(test_matrix.shape[1]):
        res = find_nodes_for_each_edge(test_matrix[:, column])  # res represents the index of nodes.
        res_sum = [0 for i in range(test_matrix.shape[1])]  # compute neighbor hyperedges of a specific hyperedge.
        for i in range(len(res)):
            res_sum = np.sum([res_sum, test_matrix[res[i], :]], axis=0).tolist()
        hyperedge_neighbor = [i for i, e in enumerate(res_sum) if e != 0]
        hyperedge_neighbor.remove(column)
        edge_neighbor.append(hyperedge_neighbor)
    return edge_neighbor


def get_cosine_angle(p3, p4):  # add 4.4
    '''
    use two speed to get the cosine angles
    '''
    m1 = math.sqrt(p3[0] ** 2 + p3[1] ** 2)
    m2 = math.sqrt(p4[0] ** 2 + p4[1] ** 2)
    m = p3[0] * p4[0] + p3[1] * p4[1]
    if m1 == 0 or m2 == 0:
        return 0
    return m / (m1 * m2)


def get_sin(p1, p2, p3):  # add 4.4
    '''
    use three pos to get the sin between two vectors
    '''
    p4 = np.zeros(2)
    p4[0] = p2[0] - p1[0]
    p4[1] = p2[1] - p1[1]
    m1 = math.sqrt(p4[0] ** 2 + p4[1] ** 2)  # |lij|
    m2 = math.sqrt(p3[0] ** 2 + p3[1] ** 2)  # |u|
    m = np.cross((p4[0], p4[1]), (p3[0], p3[1]))
    if m2 == 0:
        return 0
    return m / (m1 * m2)  # get the cosine value of eq1


def get_cosine(p1, p2, p3):
    '''
    use three pos to get the cosine between two vector
    '''
    p4 = np.zeros(2)
    p4[0] = p2[0] - p1[0]
    p4[1] = p2[1] - p1[1]
    m1 = math.sqrt(p4[0] ** 2 + p4[1] ** 2)  # |lij|
    m2 = math.sqrt(p3[0] ** 2 + p3[1] ** 2)  # |u|
    m = p4[0] * p3[0] + p4[1] * p3[1]  # vector lij multiple vector u
    if m2 == 0:
        return 0
    return m / (m1 * m2)  # get the cosine value of eq1


def anorm(p1, p2, p3, p4):
    cosine_ij = get_cosine(p1, p2, p3)
    vi_norm = get_norm(p3)
    cosine_ji = get_cosine(p2, p1, p4)
    vj_norm = get_norm(p4)
    dis = get_distance(p1, p2)
    norm = (vi_norm * cosine_ij + vj_norm * cosine_ji) / dis  # the part of eq 1 in paper
    return norm


def judge_matrix_zero(test_matrix):
    return not (np.any(test_matrix))


def compute_hypernodes_weight(test_matrix):  # add 4.4
    '''
    compute the weights of hypernodes
    '''
    hypernodes_weight_dict = {}
    for i in range(test_matrix.shape[0]):
        for j in range(i + 1, test_matrix.shape[0]):
            res = test_matrix[i, :] + test_matrix[j, :]
            key = (i, j)
            common_nodes = [i for i, e in enumerate(res) if e == 2]
            hypernodes_weight_dict[key] = len(common_nodes)
    return hypernodes_weight_dict


def find_hypernode_weight(node_weight_dict, node_i, node_j):  # add 4.4
    '''
    find the weights of hypernodes
    '''
    if node_i > node_j:
        hypernode_weight_j_i = node_weight_dict[(node_j, node_i)]
        return hypernode_weight_j_i
    else:
        hypernode_weight_i_j = node_weight_dict[(node_i, node_j)]
        return hypernode_weight_i_j


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def social_soft_attention(seq_, seq_rel, norm_lap_matr=True, qz=0.15):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]  # observed_length or predicted_length
    max_nodes = seq_.shape[0]  # npeds

    V = np.zeros((seq_len, max_nodes, 2))  # [seq_len, npeds, 2]
    fssa_weight = np.zeros((seq_len, max_nodes, max_nodes))  # [seq_len, npeds, npeds]
    for s in range(seq_len):  # observed_length or predicted_length
        step_ = seq_[:, :, s]  # [npeds, 2, observed_length] or [npeds, 2, predicted_length]
        step_rel = seq_rel[:, :, s]  # [npeds, 2, observed_length] or [npeds, 2, predicted_length]

        # initialize a incidence matrix
        incidence_matrix = np.zeros((max_nodes, max_nodes))  # add 4.4
        distance_matrix = np.zeros((max_nodes, max_nodes))  # add 4.14
        juli_weight_matrix = np.zeros((max_nodes, max_nodes))  # add 4.15

        for h in range(len(step_)):  # Traverse each pedestrian ID
            V[s, h, :] = step_rel[h]  # step_rel means the value of ped i between time t and time t-1
            incidence_matrix[h, h] = 1.0
            distance_matrix[h, h] = 1.0
            juli_weight_matrix[h, h] = 1.0
            for k in range(len(step_)):
                if k == h:
                    continue
                l2_norm_distance = get_distance(step_[k], step_[h])
                distance_matrix[k, h] = l2_norm_distance
                cosine_a = get_cosine(step_[h], step_[k], step_rel[h])
                cosine_b = get_cosine(step_[k], step_[h], step_rel[k])
                cosine_theta = get_cosine_angle(step_rel[h], step_rel[k])
                sine_a = get_sin(step_[h], step_[k], step_rel[h])
                sine_b = get_sin(step_[k], step_[h], step_rel[k])
                if ((0 <= cosine_a <= 1) and (-cosine_a <= cosine_b <= 0) and (
                        0 <= cosine_theta <= 1)) or ((0 <= cosine_a <= 1) and (0 <= cosine_b <= sine_a) and (
                        0 <= cosine_theta <= sine_a)) or ((0 <= cosine_a <= 1) and (sine_a <= cosine_b <= 1) and (
                        -cosine_a <= cosine_theta <= 0)) or ((0 <= cosine_a <= 1) and (cosine_a <= cosine_b <= 1) and (
                        -1 <= cosine_theta <= -cosine_a)):
                    if (0 <= cosine_a <= 1) and (cosine_a <= cosine_b <= 1) and (-1 <= cosine_theta <= -cosine_a):
                        incidence_matrix[k, h] = 1.0
                        juli_weight_matrix[k, h] = 3.0 / l2_norm_distance

                    if (0 <= cosine_a <= 1) and (sine_a <= cosine_b <= 1) and (-cosine_a <= cosine_theta <= 0):
                        incidence_matrix[k, h] = 1.0
                        juli_weight_matrix[k, h] = 3.0 / l2_norm_distance

                    if (0 <= cosine_a <= 1) and (0 <= cosine_b <= sine_a) and (0 <= cosine_theta <= sine_a):
                        juli_weight_matrix[k, h] = 2.0 / l2_norm_distance
                        incidence_matrix[k, h] = 1.0

                    if (0 <= cosine_a <= 1) and (-cosine_a <= cosine_b <= 0) and (0 <= cosine_theta <= 1):
                        juli_weight_matrix[k, h] = 1.0 / l2_norm_distance
                        incidence_matrix[k, h] = 1.0

        chao_bian_list = np.sum(juli_weight_matrix, axis=0)

        # compute hyperedges' degree matrix
        hyperedge_degree = compute_hyperedges_degree(incidence_matrix)
        ni_hyperedge_degree_matrix = np.diag(np.power(hyperedge_degree, -1).flatten())

        # # compute hypernodes' degree matrix
        hypernodes_degree = compute_hypernodes_degree(incidence_matrix)
        ni_hypernodes_degree_matrix = np.diag(np.power(hypernodes_degree, -0.5).flatten())

        # add 1 to zero element in chao_bian_list
        weight_list = [x if x != 0.0 else 1.0 for x in chao_bian_list]
        # get the newly weight matrix
        weight_matrix = [weight_list[i] / sum(weight_list) for i in range(len(weight_list))]

        # compute adjacent matrix
        incidence_matrix_T = incidence_matrix.T
        adjacent_matrix = compute_adjacent_matrix(incidence_matrix, weight_matrix, ni_hyperedge_degree_matrix,
                                                  incidence_matrix_T)
        adjacent_matrix_softmax = softmax_operation(adjacent_matrix)

        if norm_lap_matr:  # norm_lap_matr is True, so this code will running!
            fssa_weight[s] = ni_hypernodes_degree_matrix.dot(adjacent_matrix_softmax).dot(ni_hypernodes_degree_matrix)

    # the nodes information of GCN: the speed and weight
    return torch.from_numpy(V).type(torch.float), \
           torch.from_numpy(fssa_weight).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)  # t=[0,1,2,...,traj_len-1]
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
            min_ped=1, delim='\t', norm_lap_matr=True):
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        # all_files is a list which include all .txt files in data_dir. 
        all_files = [os.path.join(data_dir, path) for path in os.listdir(data_dir) if
                     path[0] != "." and path.endswith(".txt")]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        fet_map = {}
        fet_list = []

        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            hkl_path = os.path.splitext(path)[0] + ".pkl"
            with open(hkl_path, 'rb') as handle:
                new_fet = pickle.load(handle)
            fet_map[hkl_path] = torch.from_numpy(new_fet)
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):  # for each peds, we gain the x,y coordinates
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:  # Delete pedestrians whose observation length is not 20
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])  # Dimension Conversion
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    # we should know the shape of curr_ped_seq
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq  # copy original xy-coordinates into curr_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq  # store the relative xy-coordinaats
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    fet_list.append(hkl_path)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        self.fet_map = fet_map
        self.fet_list = fet_list

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()  # add element 0 in the front of list.
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        # Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            start, end = self.seq_start_end[ss]
            # we should know the shape of self.obs_traj_rel[start:end, :]
            v_, a_ = social_soft_attention(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :],
                                           self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_, a_ = social_soft_attention(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :],
                                           self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq  # for eth, the value is 2785

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index],
            self.fet_map[self.fet_list[index]]
        ]
        return out
