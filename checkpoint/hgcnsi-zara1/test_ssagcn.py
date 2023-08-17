from __future__ import division
import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import *
from metrics import *
from model import SocialSoftAttentionGCN
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import time
import networkx as nx
import warnings
from icecream import ic
import functools

warnings.filterwarnings('ignore')


def resampling_process(listname, n):
    ran_w = np.random.rand(n)  # 产生N个随机数
    dd = [0 for i in range(n)]
    for i in range(len(ran_w)):
        j = 0
        while ran_w[i] > listname[j]:  # 若随机数在区间之内，则将下标(j+1)存入dd中；listname中存储的是粒子的权重
            if j < n - 1:
                if ran_w[i] <= listname[j + 1]:
                    break
                else:
                    j += 1
            else:
                j = j - 1
                break
        dd[i] = j + 1
    return dd


def test(KSTEPS=20, scale=0.05):
    global loader_test, model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step = 0
    for batch in loader_test:
        step += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        # for eth_test.pt
        # obs_traj.shape: torch.Size([1, 2, 2, 8])
        # obs_traj:[1,npeds,2,observe_length]

        # pred_traj_gt.shape: torch.Size([1, 2, 2, 12])
        # pred_traj_gt:[1,npeds,2,predicted_length]

        # obs_traj_rel.shape: torch.Size([1, 2, 2, 8])
        # obs_traj_rel:[1,npeds,2,observe_length]

        # pred_traj_gt_rel.shape: torch.Size([1, 2, 2, 12])
        # pred_traj_gt_rel:[1,npeds,2,predicted_length]

        # non_linear_ped.shape: torch.Size([1, 2])
        # non_linear_ped:[1,npeds]

        # loss_mask.shape: torch.Size([1, 2, 20])
        # loss_mask:[1,npeds,20]

        # V_obs.shape: torch.Size([1, 8, 2, 2])
        # V_obs:[1,observe_length,npeds,2]

        # A_obs.shape: torch.Size([1, 8, 2, 2])
        # A_obs:[1,observe_length,npeds,npeds]

        # V_tr.shape: torch.Size([1, 12, 2, 2])
        # V_tr:[1,predicted_length,npeds,2]

        # A_tr.shape: torch.Size([1, 12, 2, 2])
        # A_tr:[1,predicted_length,npeds,npeds]

        # vgg_list.shape: torch.Size([1, 14, 14, 512])
        # for eth_test.pt, all shape are torch.Size([1, 14, 14, 512])

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr, vgg_list = batch
        obs_traj *= scale
        pred_traj_gt *= scale
        obs_traj_rel *= scale
        pred_traj_gt_rel *= scale
        V_obs *= scale
        V_tr *= scale
        num_of_objs = obs_traj_rel.shape[1]  # 2

        # V_obs.shape:torch.Size([1, 8, 2, 2])
        # V_obs_tmp.shape:torch.Size([1, 2, 8, 2])
        # V_obs:[1, observe_length, npeds, 2]
        # V_obs_tmp:[1, 2, observe_length, npeds]
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        # V_obs_tmp.shape:torch.Size([1, 2, 8, 2])
        # A_obs.squeeze().shape:torch.Size([8, 2, 2])
        # A_obs.squeeze().shape:[observe_length,npeds,npeds]
        # vgg_list.shape:torch.Size([1, 14, 14, 512])
        # V_pred.shape:torch.Size([1,5,12,npeds])
        V_pred, _ = model(V_obs_tmp, A_obs.squeeze(), vgg_list)

        # V_pred.shape:torch.size([1,12,npeds,5])
        V_pred = V_pred.permute(0, 2, 3, 1)

        # V_tr.shape: torch.Size([1, 12, 2, 2])
        # V_tr.shape: torch.Size([12, 2, 2])
        V_tr = V_tr.squeeze()

        # A_tr.shape: torch.Size([1, 12, 2, 2])
        # A_tr.shape: torch.Size([12, 2, 2])
        A_tr = A_tr.squeeze()

        # V_pred.shape:torch.size([1,12,npeds,5])
        # V_pred.shape:torch.size([12,npeds,5])
        V_pred = V_pred.squeeze()

        # obs_traj_rel.shape: torch.Size([1, 2, 2, 8])
        # num_of_objs=npeds, 2
        num_of_objs = obs_traj_rel.shape[1]

        # V_pred.shape:torch.size([12,npeds,5])
        # V_tr:[12,npeds,2]
        # V_pred.shape: torch.Size([12, 2, 5])
        # V_tr.shape: torch.Size([12, 2, 2])
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        # mux.shape:torch.Size([12, npeds])
        # muy.shape:torch.Size([12, npeds])
        # sx.shape:torch.Size([12, npeds])
        # sy.shape:torch.Size([12, npeds])
        # corr.shape:torch.Size([12, npeds])
        mux = V_pred[:, :, 0]  # mux
        muy = V_pred[:, :, 1]  # muy
        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr
        mux_deepcopy = mux.clone()
        muy_deepcopy = muy.clone()
        sx_deepcopy = sx.clone()
        sy_deepcopy = sy.clone()
        corr_deepcopy = corr.clone()

        # V_pred.shape: torch.Size([12, 2, 5])
        # cov.shape: torch.Size([12, 2, 2,2])
        # cov.shape: [predicted_length, npeds, 2, 2]
        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()

        # the dimension of "sx * sx","corr * sx * sy","corr * sx * sy" and "sy * sy" are all [predicted_length, npeds]
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        # mean.shape: [predicted_length, npeds, 2]
        mean = V_pred[:, :, 0:2]

        # mean.shape: [predicted_length, npeds, 2]
        # cov.shape: [predicted_length, npeds, 2, 2]
        mvnormal = torchdist.MultivariateNormal(mean, cov)

        ### Rel to abs
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len

        # Now sample 20 samples
        ade_ls = {}
        fde_ls = {}

        # obs_traj.shape: torch.Size([1, 2, 2, 8])
        # obs_traj.shape: [1, npeds, 2, observe_length]
        # V_obs.shape: torch.Size([1, 8, 2, 2])
        # V_x.shape: torch.Size([8, 2, 2])
        # V_x.shape: [observe_length, npeds, 2]
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())

        # V_obs:[1, observe_length, npeds, 2]
        # V_obs.shape: torch.Size([1, 8, 2, 2])
        # V_x[0, :, :].shape: [npeds, 2]
        # V_x[0, :, :].shape: [2, 2]
        # squeeze() in numpy just delete single dimension
        # V_x_rel_to_abs.shape:[8, 2, 2]
        # V_x_rel_to_abs.shape:[8, npeds, 2]
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(), V_x[0, :, :].copy())

        # pred_traj_gt.shape: torch.Size([1, 2, 2, 12])
        # pred_traj_gt:[1,npeds,2,predicted_length]
        # V_y.shape: torch.Size([12, 2, 2])
        # V_y.shape: [predicted_length, npeds, 2]
        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())

        # V_tr.shape: torch.Size([12, 2, 2])
        # V_tr: [12, npeds, 2]
        # V_x[-1, :, :].shape: [npeds, 2]
        # V_x[-1, :, :].shape: [2, 2]
        # V_x[0, :, :] represents the first element of V_x
        # V_x[-1, :, :] represents the last element of V_x
        # V_y_rel_to_abs.shape:[12, 2, 2]
        # V_y_rel_to_abs.shape:[12, npeds, 2]
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(), V_x[-1, :, :].copy())

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):  # traversing the id of each pedestrian
            ade_ls[n] = []
            fde_ls[n] = []
        # ic(step)
        # cnt_tian = 0
        for k in range(KSTEPS):  # the value of KSTEPS is 20
            # cnt_tian += 1
            # ic(cnt_tian)
            V_pred = mvnormal.sample()  # V_pred.shape:[predicted_length,npeds,2]
            # # # take particle filter measure!
            # particle_number = 10  # represents the number of particles
            # filter_value = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2)
            # weight_cluster_ = torch.zeros(particle_number, V_pred.shape[1], 2)
            # cum_ = torch.zeros(particle_number, V_pred.shape[1], 2)
            # for pred_len in range(filter_value.shape[0]):
            #     o_mux = mux_deepcopy[pred_len, :].data.cpu()
            #     o_muy = muy_deepcopy[pred_len, :].data.cpu()
            #     o_sx = sx_deepcopy[pred_len, :].data.cpu()
            #     o_sy = sy_deepcopy[pred_len, :].data.cpu()
            #     o_corr = corr_deepcopy[pred_len, :].data.cpu()
            #     predicted_x = torch.zeros(filter_value.shape[1])  # represents the npeds
            #     predicted_y = torch.zeros(filter_value.shape[1])
            #     next_values_cluster = torch.zeros(filter_value.shape[1], particle_number, 2)  # the first is npeds
            #     for node in range(filter_value.shape[1]):
            #         mmean = [o_mux[node], o_muy[node]]
            #         ccov = [[o_sx[node] * o_sx[node], o_corr[node] * o_sx[node] * o_sy[node]],
            #                 [o_corr[node] * o_sx[node] * o_sy[node], o_sy[node] * o_sy[node]]]
            #         mmean = np.array(mmean, dtype='float')  # tensor -> numpy
            #         ccov = np.array(ccov, dtype='float')
            #         predicted_value = np.random.multivariate_normal(mmean, ccov, particle_number)
            #         predicted_x[node] = predicted_value[0][0]
            #         predicted_y[node] = predicted_value[0][1]
            #         next_values_cluster[node, :, 0] = torch.from_numpy(predicted_value[:, 0])
            #         next_values_cluster[node, :, 1] = torch.from_numpy(predicted_value[:, 1])
            #     next_values_cluster_copy = next_values_cluster.clone()
            #     particle_main = torch.zeros(filter_value.shape[1], 2)
            #     for i in range(filter_value.shape[1]):
            #         particle_main_i = torch.from_numpy(np.mean(next_values_cluster.clone().numpy()[i, :], axis=0))
            #         particle_main[i] = particle_main_i
            #     for i in range(next_values_cluster.shape[1]):
            #         particle_i = next_values_cluster[:, i]
            #         weight_cluster_[i] = 1 / (np.sqrt(2 * np.pi * (1 / particle_number))) * np.exp(
            #             -(particle_main - particle_i) ** 2 / (2 * (1 / particle_number)))
            #     weight_cluster_ = weight_cluster_ / (sum(weight_cluster_) + 1e-20)
            #     for j in range(next_values_cluster.shape[1]):
            #         cum_[j] = functools.reduce(lambda x, y: x + y, weight_cluster_[:j + 1])
            #     for i in range(filter_value.shape[1]):
            #         i_x = resampling_process(cum_[:, i, 0], particle_number)  # the second number is particle number
            #         i_y = resampling_process(cum_[:, i, 1], particle_number)
            #         next_values_cluster_copy[i, [k for k in range(particle_number)], 0] = next_values_cluster[
            #             i, i_x, 0]
            #         next_values_cluster_copy[i, [k for k in range(particle_number)], 1] = next_values_cluster[
            #             i, i_y, 1]
            #     next_values_cluster_copy = next_values_cluster_copy.mean(axis=1, keepdim=False)
            #     new_x = next_values_cluster_copy[:, 0]
            #     new_y = next_values_cluster_copy[:, 1]
            #     filter_value[pred_len, :, 0] = new_x
            #     filter_value[pred_len, :, 1] = new_y
            # V_pred_rel_to_abs = nodes_rel_to_nodes_abs(filter_value.data.cpu().numpy().squeeze().copy(),
            #                                            V_x[-1, :, :].copy())
            # take particle filter measure!

            # V_pred_rel_to_abs.shape:[12, npeds, 2]
            # V_pred_rel_to_abs represents the predicted trajectories, it will generate 12 predicted trajectories.
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(), V_x[-1, :, :].copy())

            # draw graph
            # edge_colors=[]
            # Vtr=pred_traj_gt.squeeze(dim=0).permute(2,0,1).data.cpu().numpy()[0]
            # adj=A_obs.squeeze(dim=0).data.cpu().numpy()[-1]
            # for i in adj.flatten() :
            #     if i>0:
            #         # i=np.ceil(i)*30
            #         edge_colors.append(i)
            # adj_one=adj/adj
            # adj_one[np.isnan(adj_one)]=0
            # G = nx.from_numpy_matrix(adj_one)
            # pos=Vtr
            # for u,v,d in G.edges(data=True):
            #     d['weight'] = adj[u][v]
            # plt.title("Batch:"+str(step))
            # edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
            # # nodes = nx.draw_networkx_nodes(G, pos, node_size=100, node_color='red')
            # edgess = nx.draw_networkx_edges(G, pos, node_size=100,
            #                    arrowsize=10, edge_color=weights,
            #                    edge_cmap=plt.cm.BuGn, width=2)
            # nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights, width=10.0, edge_cmap=plt.cm.BuPu)
            # ax = plt.gca()
            # ax.set_axis_off()
            # plt.colorbar(edgess)
            # for i in range(obs_traj.shape[1]) :
            #     obs_traj=torch.cat((obs_traj,pred_traj_gt[:,:,:,0].unsqueeze(dim=3)),3)   # NVCT
            #     plt.plot(obs_traj[0,i,0,:].cpu().numpy(), obs_traj[0,i,1,:].cpu().numpy(), 'm.-.',color='blue', label='training accuracy')
            #     plt.plot(pred_traj_gt[0,i,0,:].cpu().numpy(), pred_traj_gt[0,i,1,:].cpu().numpy(), 'm.-.',color='red', label='training accuracy')

            # plt.show()

            colors = list(['Blues', 'Greens', 'Oranges', 'Reds', 'PuRd', 'PuBu', 'Purples'])

            if step == 241:
                for i in range(obs_traj.shape[1]):  # traversing the id of each pedestrian
                    # ax = sns.kdeplot(x=V_pred_rel_to_abs[:,i,0],y=V_pred_rel_to_abs[:,i,1],shade = True, cmap = colors[i%6])
                    # V_pred_rel_to_abs[:, i, 0] means the x coordinates
                    # V_pred_rel_to_abs[:, i, 1] means the y coordinates
                    ax = sns.kdeplot(x=V_pred_rel_to_abs[:, i, 0], y=V_pred_rel_to_abs[:, i, 1], fill=True,
                                     cmap=colors[i % 6])
                    ax.patch.set_facecolor('white')
                    ax.collections[0].set_alpha(0)
                    obs_traj = torch.cat((obs_traj, pred_traj_gt[:, :, :, 0].unsqueeze(dim=3)), 3)  # NVCT
                    # plt.plot(obs_traj[0,i,0,:].cpu().numpy(), obs_traj[0,i,1,:].cpu().numpy(), 'm.-.',color='blue', label='training accuracy')
                    # plt.plot(pred_traj_gt[0,i,0,:].cpu().numpy(), pred_traj_gt[0,i,1,:].cpu().numpy(), 'm.-.',color='red', label='training accuracy')
                # plt.title("Batch:"+str(step))
                # plt.show()

            # Copy the predicted values, and put it into list
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

            for n in range(num_of_objs):  # traversing the id of each pedestrian
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])  # predicted trajectories,the time length is 12
                target.append(V_y_rel_to_abs[:, n:n + 1, :])  # real trajectories,the time length is 12
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])  # real trajectories,the time length is 8
                number_of.append(1)  # lists used for delimiter

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        for n in range(num_of_objs):  # traversing the id of each pedestrian
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    # Calculated the smallest ADE value of 20 samples per pedestrian, deposited in the list;
    # #Finally, average all elements in the list to get the overall average ADE loss value
    ade_ = sum(ade_bigls) / len(ade_bigls)

    # Calculated the smallest FDE value of 20 samples per pedestrian, deposited in the list;
    # #Finally, average all elements in the list to get the overall average FDE loss value
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_, fde_, raw_data_dict


if __name__ == '__main__':
    paths = ['./checkpoint/*ssagcn*']  # paths=['./checkpoint/*ssagcn*']
    # KSTEPS = 20
    KSTEPS = 20
    print("*" * 50)
    print('Number of samples:', KSTEPS)
    print("*" * 50)

    for feta in range(len(paths)):
        ade_ls = []
        fde_ls = []
        path = paths[feta]  # path='./checkpoint/*ssagcn*'
        exps = glob.glob(path)
        print('Model being tested are:',
              exps)  # exps=['./checkpoint\\ssagcn-eth', './checkpoint\\ssagcn-hotel', './checkpoint\\ssagcn-univ', './checkpoint\\ssagcn-zara1', './checkpoint\\ssagcn-zara2']
        cnt_cnt = 0
        for exp_path in exps:
            cnt_cnt += 1
            if cnt_cnt == 1 or cnt_cnt == 2 or cnt_cnt == 3:
                continue
            print("*" * 50)
            print("Evaluating model:", exp_path)

            # verify train_best.pth add 2023.4.11
            # model_path = exp_path + '/train_best.pth'
            model_path = exp_path + '/val_best.pth'

            args_path = exp_path + '/args.pkl'
            with open(args_path, 'rb') as f:
                args = pickle.load(f)  # args already be copied in .txt file called 'args'

            scale = args.scale
            # verify train_best.pth add 2023.4.11
            # stats = exp_path + '/constant_metrics_train.pkl'
            stats = exp_path + '/constant_metrics_val.pkl'

            with open(stats, 'rb') as f:
                cm = pickle.load(f)
            print("Stats:", cm)

            # Data prep
            obs_seq_len = args.obs_seq_len
            pred_seq_len = args.pred_seq_len

            dset_test = torch.load("./data/" + args.dataset + "_test.pt")

            print("data %s load complete" % args.dataset)
            # dset_test = TrajectoryDataset(
            #         data_set+'test/',
            #         obs_len=obs_seq_len,
            #         pred_len=pred_seq_len,
            #         skip=1,norm_lap_matr=True)

            loader_test = DataLoader(
                dset_test,
                batch_size=1,
                shuffle=False,
                num_workers=1)

            # Defining the model
            # args.n_ssagcn=1
            # args.n_txpcnn=7
            # args.output_size=5
            # args.obs_seq_len=8
            # args.pred_seq_len=12
            # args.kernel_size=3
            model = SocialSoftAttentionGCN(stgcn_num=args.n_ssagcn, tcn_num=args.n_txpcnn,
                                           output_feat=args.output_size, seq_len=args.obs_seq_len,
                                           kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).cuda()
            model.load_state_dict(torch.load(model_path))

            ade_ = 999999
            fde_ = 999999
            print("Testing ....")
            time_start = time.time()
            ad, fd, raw_data_dic_ = test(KSTEPS, scale)
            time_end = time.time()
            ic(time_end - time_start)
            ade_ = min(ade_, ad) / scale
            fde_ = min(fde_, fd) / scale
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            print("ADE:", ade_ / scale, " FDE:", fde_ / scale)
            if cnt_cnt == 4:
                break

        print("*" * 50)
        print("Avg ADE:", sum(ade_ls) / len(ade_ls) / scale)
        print("Avg FDE:", sum(fde_ls) / len(ade_ls) / scale)
