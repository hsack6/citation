import torch
import numpy as np
import scipy
import random

import torch.nn as nn
import matplotlib.pyplot as plt
import math
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import date
import scipy.sparse as sp
import itertools
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
import os
import pickle

class Data:
    x = None
    edge_index =None
    anchorset_id = None
    dists_max = None
    dists_argmax = None
    dists = None
    def __init__(self, x, edge_index, dists_max = None, dists_argmax = None, dists = None):
        self.x = x
        self.edge_index = edge_index
        self.dists_max = dists_max
        self.dists_argmax = dists_argmax
    def copy(self):
        return Data(self.x, self.edge_index,
                    self.dists_max if not self.dists_max is None else None,
                    self.dists_argmax if not self.dists_argmax is None else None,
                    self.dists if not self.dists is None else None)

def score_link_prediction(labels, scores):
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)

def get_pred(cmodel, nodes, gt_labels, X, nodes_keep, lambdas = (0,1,1)):
    # anode_emb = torch.sparse.mm(data.x, cmodel.attr_emb(torch.arange(data.x.shape[1]).to(cmodel.device)))
    test_data = Data(X, None)
    anode_emb = cmodel.attr_emb(test_data)

    first_embs = anode_emb[nodes[:,0]]

    sec_embs = anode_emb[nodes[:,1]]
    res = cmodel.attr_layer(first_embs,sec_embs) * lambdas[1]

    node_emb = anode_emb.clone()

    res = res + cmodel.inter_layer(first_embs,node_emb[nodes[:,1]]) * lambdas[2]

    if len(res.shape)>1:
        res = res.softmax(dim=1)[:,1]
    res = res.detach().cpu().numpy()
    return gt_labels, res

def ind_eval(cmodel, nodes, gt_labels,X,nodes_keep, lambdas = (0,1,1)):
    """
    :param cmodel: pytorch model
    :param nodes: (2873, 2) stack(test_ones, test_zeros)
    :param first_emb: (2873, 64)
    :param sec_emb: (2873, 64)
    :param res: (2873) pred_score
    :param gt_labels: (2873, )
    :param X:2
    :param nodes_keep:
    :param lambdas:
    :return:
    """
    # anode_emb = torch.sparse.mm(data.x, cmodel.attr_emb(torch.arange(data.x.shape[1]).to(cmodel.device)))
    test_data = Data(X, None)
    anode_emb = cmodel.attr_emb(test_data)

    first_embs = anode_emb[nodes[:,0]]

    sec_embs = anode_emb[nodes[:,1]]
    res = cmodel.attr_layer(first_embs,sec_embs) * lambdas[1]

    node_emb = anode_emb.clone()

    res = res + cmodel.inter_layer(first_embs,node_emb[nodes[:,1]]) * lambdas[2]

    if len(res.shape)>1:
        res = res.softmax(dim=1)[:,1]
    res = res.detach().cpu().numpy()
    return score_link_prediction(gt_labels, res)

def tran_eval(cmodel, test_data, gt_labels,data, lambdas = (1,1,1)):
    res = cmodel.evaluate(test_data, data, lambdas)
    if len(res.shape)>1:
        res = res.softmax(dim=1)[:,1]
    res = res.detach().cpu().numpy()
    return score_link_prediction(gt_labels, res)

def rprint(s):
    s = str(s)
    print('\r'+s+"",end='')

def get_delta(edge_index,A):
    a,b= np.unique(edge_index,return_counts=True)
    order = a[np.argsort(b)[::-1]]
    delta = torch.zeros(A.shape[0])
    delta[a] = torch.FloatTensor(b/b.max())
    return delta

def get_train_data(A_train, batch_size, tv_edges, inductive):
    nodes = []
    labels = []
    tmp_A = A_train.tolil()
    nodeNum = A_train.shape[0]

    if not inductive:
        forbidden_Matrix = sp.lil_matrix(A_train.shape)
        forbidden_Matrix[tv_edges[:, 0], tv_edges[:, 1]] = 1
        while True:
            a = random.randint(0, nodeNum - 1)
            b = random.randint(0, nodeNum - 1)
            if not (forbidden_Matrix[a, b]):
                nodes.append([a, b])
                if tmp_A[a, b]:
                    labels.append(1)
                else:
                    labels.append(0)

            if len(tmp_A.rows[a]):
                neigh = np.random.choice(tmp_A.rows[a])
                if not (forbidden_Matrix[a, neigh]):
                    nodes.append([a, neigh])
                    labels.append(1)

            if len(labels) >= batch_size:
                yield torch.LongTensor(nodes), torch.LongTensor(labels)
                del nodes[:]
                del labels[:]
    else:
        while True:
            a = random.randint(0, nodeNum - 1)
            b = random.randint(0, nodeNum - 1)
            nodes.append([a, b])
            if tmp_A[a, b]:
                labels.append(1)
            else:
                labels.append(0)

            if len(tmp_A.rows[a]):
                neigh = np.random.choice(tmp_A.rows[a])
                nodes.append([a, neigh])
                labels.append(1)

            if len(labels) >= batch_size:
                yield torch.LongTensor(nodes), torch.LongTensor(labels)
                del nodes[:]
                del labels[:]

def get_train_inputs(data, test_edges, val_edges, batch_size, neg_sample_num=10, undirected=True, inductive=False):
    test_mask = (1 - torch.eye(data.dists.shape[0])).bool()
    if not inductive:
        test_mask[test_edges[:, 0], test_edges[:, 1]] = 0
        test_mask[val_edges[:, 0], val_edges[:, 1]] = 0
        if undirected:
            test_mask[test_edges[:, 1], test_edges[:, 0]] = 0
            test_mask[val_edges[:, 1], val_edges[:, 0]] = 0
    test_mask = test_mask.to(data.dists.device)
    filter_dists = data.dists * test_mask
    pos = (filter_dists == 0.5).nonzero()
    filter_dists[pos[:, 0], pos[:, 1]] = 0
    pos = pos.cpu().tolist()
    pos_dict = {}
    for i, j in pos:
        pos_dict[i] = pos_dict.get(i, []) + [j]

    neg_dict = {}
    neg = (filter_dists > 0.12).nonzero().cpu().tolist()
    for i, j in neg:
        neg_dict[i] = neg_dict.get(i, []) + [j]
    nodes = list(pos_dict.keys())
    random.shuffle(nodes)
    inputs = []
    while True:
        for node in nodes:
            tmp_imput = [node, pos_dict[node], random.sample(neg, neg_sample_num) if len(neg) > neg_sample_num else neg]
            inputs.append(tmp_imput)
            if len(inputs) >= batch_size:
                yield np.array(inputs)
                del inputs[:]
        random.shuffle(nodes)

def get_random_anchorset(n,c=0.5):
    m = int(np.log2(n))
    copy = int(c*m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n,size=anchor_size,replace=False))
    return anchorset_id

def get_dist_max(anchorset_id, dist, device):
    dist_max = torch.zeros((dist.shape[0],len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0],len(anchorset_id))).long().to(device)
    for i in range(len(anchorset_id)):
        temp_id = anchorset_id[i]
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:,i] = dist_max_temp
        # dist_argmax[:,i] = dist_argmax_temp
        dist_argmax[:,i] = torch.LongTensor(temp_id).to(device)[dist_argmax_temp]
    return dist_max, dist_argmax

def preselect_anchor(data, layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu'):

    # data.anchor_size_num = anchor_size_num
    # data.anchor_set = []
    # anchor_num_per_size = anchor_num//anchor_size_num
    # for i in range(anchor_size_num):
    #     anchor_size = 2**(i+1)-1
    #     anchors = np.random.choice(data.num_nodes, size=(layer_num,anchor_num_per_size,anchor_size), replace=True)
    #     data.anchor_set.append(anchors)
    # data.anchor_set_indicator = np.zeros((layer_num, anchor_num, data.num_nodes), dtype=int)

    anchorset_id = get_random_anchorset(data.x.shape[0],c=1)
    data.anchorset_id = anchorset_id
    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)

def convert_sSp_tSp(x):
    coo = x.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def get_hops(A, K):
    """
    Calculates the K-hop neighborhoods of the nodes in a graph.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        The graph represented as a sparse matrix
    K : int
        The maximum hopness to consider.

    Returns
    -------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
    """
    hops = {1: A.tolil(), -1: A.tolil()}
    hops[1].setdiag(0)

    for h in range(2, K + 1):
        # compute the next ring
        next_hop = hops[h - 1].dot(A)
        next_hop[next_hop > 0] = 1

        for prev_h in range(1, h):
            next_hop -= next_hop.multiply(hops[prev_h])

        next_hop = next_hop.tolil()
        next_hop.setdiag(0)

        hops[h] = next_hop
        hops[-1] += next_hop

    return hops

def load_datafile(data_arrays_link, dists, ind_train_A, ind_train_X, nodes_keep, A, X, args):
    device = torch.device('cuda:' + str(args.cuda) if args.gpu else 'cpu')
    A = scipy.sparse.csc_matrix(A)
    X = scipy.sparse.csc_matrix(X)
    train_ones, val_ones, val_zeros, test = data_arrays_link[0][0].numpy(), data_arrays_link[1][0].numpy(), data_arrays_link[2][0].numpy(), data_arrays_link[3][0].numpy()

    A_train = scipy.sparse.csc_matrix(ind_train_A)
    X_train = scipy.sparse.csc_matrix(ind_train_X)

    hops = get_hops(A_train, 1)

    scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                       hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                   for h in hops}

    # test_edges = np.row_stack((test_ones, test_zeros))
    test_edges = test
    val_edges = np.row_stack((val_ones, val_zeros))

    gt_labels = A[test_edges[:, 0], test_edges[:, 1]].A1
    ## test_ground_truth = torch.LongTensor(1-gt_labels) * 2

    sp_X = convert_sSp_tSp(X).to(device).to_dense()
    sp_attrM = convert_sSp_tSp(X_train).to(device)
    # us_attr_dict = get_us_attr_dict(X_train)
    val_labels = A_train[val_edges[:, 0], val_edges[:, 1]].A1

    data = Data(convert_sSp_tSp(X_train).to_dense().to(device).double(), torch.LongTensor(train_ones.T).to(device))
    data.dists = dists

    if not data.dists is None:
        data.dists = data.dists.to(device)
        preselect_anchor(data, layer_num=args.layer_num, anchor_num=64, device=device)

    return A, X, A_train, X_train, data, train_ones, val_edges, test_edges, val_labels, gt_labels, nodes_keep