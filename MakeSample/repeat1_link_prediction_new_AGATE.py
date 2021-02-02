import numpy as np
import random
import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd
import copy
import os
import sys
import glob
import re
from collections import defaultdict
import networkx as nx
from scipy.sparse import lil_matrix, coo_matrix
from scipy.io import mmwrite, mmread

# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )

from setting_param import MakeSample_link_prediction_new_InputDir as InputDir
from setting_param import Evaluation_prediction_num_of_node_new_LSTM_InputDir as predicted_num_InputDir

from setting_param import Model_attribute_prediction_new_PROSER_selecter_OutputDir as PROSER_Out_InputDir
from setting_param import Model_repeat1_attribute_prediction_new_utilize_new_attribute_link_FNN_OutputDir as AGATE_Out_InputDir
from setting_param import MakeSample_repeat1_link_prediction_new_AGATE_OutputDir as OutputDir

from setting_param import ratio_test
from setting_param import ratio_valid
from setting_param import L
from setting_param import attribute_dim

os.makedirs(OutputDir, exist_ok=True)
learning_type = ["inference"]
for l_type in learning_type:
    os.makedirs(OutputDir + "/" + l_type, exist_ok=True)
    os.makedirs(OutputDir + "/" + l_type + "/input", exist_ok=True)
    os.makedirs(OutputDir + "/" + l_type + "/label", exist_ok=True)
    os.makedirs(OutputDir + "/" + l_type + "/mask", exist_ok=True)

# READ EXIST_TABLE
EXIST_TABLE = np.load(InputDir + '/exist_table.npy')

n_node = EXIST_TABLE.shape[0]

def ExistNodeList(ts):
    assert ts >= 0, "ts < 0 [referrence error]"
    return np.where(EXIST_TABLE[:, ts]==1)[0]

def GetAppearedNodes(ts):
    return set(ExistNodeList(ts)) - set(ExistNodeList(ts-1))

def GetObservedNodes(ts, L):
    U = set()
    for i in range(L):
        U |= set(ExistNodeList(ts-i))
    return U

def GetNodes(ts, L, node_type):
    if node_type=='all':
        node_set = set(ExistNodeList(ts))
    elif node_type=='stay':
        node_set = set(ExistNodeList(ts-1)) & set(ExistNodeList(ts))
    elif node_type=='lost':
        node_set = set(ExistNodeList(ts-1)) - set(ExistNodeList(ts))
    elif node_type=='return':
        node_set = GetAppearedNodes(ts) - (GetAppearedNodes(ts) - GetObservedNodes(ts-1, L))
    elif node_type=='new':
        node_set = GetAppearedNodes(ts) - GetObservedNodes(ts-1, L)
        node_set |= GetNodes(ts, L, 'return')
    return node_set

def Nx(ts):
    return  nx.from_numpy_matrix(mmread(InputDir + '/adjacency' + str(ts)).toarray())

def SubNxNew(ts, L):
    return nx.Graph(Nx(ts).edges(GetNodes(ts, L, 'new')))

def SubNxLost(ts, L):
    return nx.Graph(Nx(ts-1).edges(GetNodes(ts, L, 'lost')))

def GetEdges(ts, L, edge_type):
    G_1 = Nx(ts)
    if edge_type == "all":
        edge_set = G_1.edges
    elif edge_type == 'stay':
        G_0 = Nx(ts - 1)
        edge_set = G_0.edges & G_1.edges
    elif edge_type == "appeared":
        G_0 = Nx(ts - 1)
        edge_set = G_1.edges - G_0.edges - SubNxNew(ts, L).edges
    elif edge_type == "disappeared":
        G_0 = Nx(ts - 1)
        edge_set = G_0.edges - G_1.edges - SubNxLost(ts, L).edges
    return edge_set

def get_adjacency_matrix(ts, L, edge_type):
    G = nx.Graph(list(GetEdges(ts, L, edge_type)))
    A = np.array(nx.to_numpy_matrix(G, nodelist=[i for i in range(n_node)]))
    return A

def get_exist_matrix(ts):
    index = np.where(EXIST_TABLE[:, ts] == 1)[0]
    exist_row = np.zeros((n_node, n_node))
    exist_row[index] = 1
    exist_col = np.zeros((n_node, n_node))
    exist_col[:, index] = 1
    return exist_row * exist_col

def NodeAttribute(ts):
    return np.load(InputDir + '/node_attribute' + str(ts) + '.npy')

def get_expanded_node_attribute_inference(ts, L, n_node, n_expanded, new):
    node_attribute = NodeAttribute(ts)
    # node_attribute[sorted(GetNodes(ts, L, 'new'))] = 0
    new_node_attribute = new

    expanded_attribute = np.zeros((n_node + n_expanded, NodeAttribute(ts).shape[1]))
    expanded_attribute[:n_node] = node_attribute
    expanded_attribute[n_node:n_node + new_node_attribute.shape[0]] = new_node_attribute
    return expanded_attribute


def get_expanded_label_matrix_inference(ts, L, expanded_idx_dic, n_node, n_expanded):
    expanded_edges = set()
    for i, j in SubNxNew(ts, L).edges:
        expanded_i = []
        if i in expanded_idx_dic.keys():
            for i_ in expanded_idx_dic[i]:
                expanded_i.append(i_)
        else:
            expanded_i.append(i)
        expanded_j = []
        if j in expanded_idx_dic.keys():
            for j_ in expanded_idx_dic[j]:
                expanded_j.append(j_)
        else:
            expanded_j.append(j)
        for i_ in expanded_i:
            for j_ in expanded_j:
                expanded_edges.add((i_, j_))
    G = nx.Graph(list(expanded_edges))
    A = np.array(nx.to_numpy_matrix(G, nodelist=[i for i in range(n_node + n_expanded)]))
    return A


def get_expanded_mask_matrix_inference(ts, L, expanded_idx_dic, n_node, n_expanded):
    expanded_matrix = np.zeros((n_node + n_expanded, n_node + n_expanded))
    for n in GetNodes(ts, L, 'new'):
        if n in expanded_idx_dic.keys():
            for s in GetNodes(ts, L, 'stay'):
                for n_ in expanded_idx_dic[n]:
                    expanded_matrix[n_][s] = 1
                    expanded_matrix[s][n_] = 1
    return expanded_matrix

def TsSplit(ts, L):
    ts_train = [(ts+l) for l in range(L)]
    ts_test = ts_train[-1]+1
    ts_all = ts_train.copy()
    ts_all.extend([ts_test])
    return ts_train, ts_test, ts_all

def load_paths_from_dir(dir_path):
    # dir 以下のファイル名のリストを取得
    path_list = glob.glob(dir_path + "/*")
    # ソート (ゼロ埋めされていない数字の文字列のソート)
    path_list = np.array(sorted(path_list, key=lambda s: int(re.findall(r'\d+', s)[-1])))
    return path_list

def dev_test_split(all_idx, n_samples, ratio_test):
    n_test = int(n_samples * ratio_test)
    return all_idx[:-n_test], all_idx[-n_test:]

def train_valid_split(dev_idx, n_samples, ratio_valid):
    n_valid = int(n_samples * ratio_valid)
    return dev_idx[:-n_valid], dev_idx[-n_valid:]

def data_split(input_dir):
    paths = load_paths_from_dir(input_dir + '/output')
    new_num_ls = []
    teacher_num_ls =[]
    teacher_idx_ls =[]
    new_ls = []
    teacher_ls = []
    node_pair_list_ls = []
    for path in paths:
        if 'new_num' in path.split('/')[-1]:
            new_num_ls.append(path)
        elif 'teacher_num' in path.split('/')[-1]:
            teacher_num_ls.append(path)
        elif 'teacher_idx' in path.split('/')[-1]:
            teacher_idx_ls.append(path)
        elif 'new' in path.split('/')[-1]:
            new_ls.append(path)
        elif 'teacher' in path.split('/')[-1]:
            teacher_ls.append(path)
        elif 'node_pair_list' in path.split('/')[-1]:
            node_pair_list_ls.append(path)
    return np.array(new_ls), np.array(teacher_ls), np.array(new_num_ls), np.array(teacher_num_ls), np.array(teacher_idx_ls), np.array(node_pair_list_ls)

def load_npy_data(new_paths, teacher_paths, new_num_paths, teacher_num_paths, teacher_idx_paths, node_pair_list_paths, all_idx, ts):
    idx = all_idx[ts-L]
    new = np.load(new_paths[idx])
    teacher = np.load(teacher_paths[idx])
    new_num = np.load(new_num_paths[idx])
    teacher_num = np.load(teacher_num_paths[idx])
    teacher_idx = np.load(teacher_idx_paths[idx])
    node_pair_list = np.load(node_pair_list_paths[idx])
    return new, teacher, new_num, teacher_num, teacher_idx, node_pair_list

def true_pred_mask_split(input_dir):
    paths = load_paths_from_dir(input_dir + '/output')
    true_ls = []
    pred_ls = []
    mask_ls = []
    for path in paths:
        if 'true' in path.split('/')[-1]:
            true_ls.append(path)
        elif 'pred' in path.split('/')[-1]:
            pred_ls.append(path)
        elif 'mask' in path.split('/')[-1]:
            mask_ls.append(path)
    return np.array(true_ls), np.array(pred_ls), np.array(mask_ls)

def load_AGATE_output(InputDir, all_idx, ts, L):
    idx = all_idx[ts-L]
    true_paths, pred_paths, mask_paths = true_pred_mask_split(InputDir)
    pred = mmread(pred_paths[idx]).toarray()
    mask = mmread(mask_paths[idx]).toarray()[0][0]
    return pred, mask

# predicted_new_node_numの最大値を取得
predicted_new_node_num_list = []
for ts in range(L, EXIST_TABLE.shape[1]-L):
    predicted_new_node_num = int(np.load(predicted_num_InputDir + '/output/pred' + str(ts) + '.npy')[0])
    predicted_new_node_num_list.append(predicted_new_node_num)
max_predicted_new_node_num = max(predicted_new_node_num_list)

# new_node_numの最大値を取得
new_node_num_list = []
for ts in range(L, EXIST_TABLE.shape[1]-L):
    ts_train, ts_test, ts_all = TsSplit(ts, L)
    new_node_num = len(GetNodes(ts_test, L, 'new'))
    new_node_num_list.append(new_node_num)
max_new_node_num = max(new_node_num_list)

n_expanded = max([max_predicted_new_node_num, max_new_node_num])

new_paths, teacher_paths, new_num_paths, teacher_num_paths, teacher_idx_paths, node_pair_list_paths = data_split(PROSER_Out_InputDir)
n_samples = len(new_paths)
all_idx = list(range(n_samples))
# dev_idx, test_idx = dev_test_split(all_idx, len(all_idx), ratio_test)
# train_idx, valid_idx = train_valid_split(dev_idx, len(all_idx), ratio_valid)
train_idx = all_idx[:-4]
valid_idx = all_idx[-4:-2]
test_idx = all_idx[-2:]

for ts in range(L, EXIST_TABLE.shape[1] - L):
    ts_train, ts_test, ts_all = TsSplit(ts, L)
    new, teacher, new_num, teacher_num, teacher_idx, node_pair_list = load_npy_data(new_paths, teacher_paths,
                                                                                    new_num_paths, teacher_num_paths,
                                                                                    teacher_idx_paths,
                                                                                    node_pair_list_paths, all_idx, ts)
    pred, mask = load_AGATE_output(AGATE_Out_InputDir, all_idx, ts, L)
    new[:mask] = pred[n_node : n_node + mask] # AGATEの予測結果に置き換える

    # reference check
    assert sorted(GetNodes(ts_test, L, 'new')) == teacher_idx.tolist()[:teacher_num], 'reference error'
    predicted_new_node_num = int(np.load(predicted_num_InputDir + '/output/pred' + str(ts) + '.npy')[0])
    assert new_num == predicted_new_node_num, 'reference error'
    new_node_num = len(GetNodes(ts_test, L, 'new'))
    assert teacher_num == new_node_num, 'reference error'
    assert new.shape[0] == max_predicted_new_node_num, 'reference error'
    assert teacher.shape[0] == max_new_node_num, 'reference error'

    # 推論用
    expanded_idx_dic = defaultdict(list)
    for new_row in range(new_num):
        teacher_node = int(node_pair_list[new_row, 1])
        new_node = int(node_pair_list[new_row, 0])
        expanded_idx_dic[teacher_node].append(n_node + new_node)
    # input
    node_attribute = get_expanded_node_attribute_inference(ts_train[-1], L, n_node, n_expanded, new)
    # label
    label = get_expanded_label_matrix_inference(ts_test, L, expanded_idx_dic, n_node, n_expanded)
    # mask
    mask = get_expanded_mask_matrix_inference(ts_test, L, expanded_idx_dic, n_node, n_expanded)

    label = lil_matrix(label)
    mask = lil_matrix(mask)

    np.save(OutputDir + "/inference/input/" + str(ts), node_attribute)
    mmwrite(OutputDir + "/inference/label/" + str(ts), label)
    mmwrite(OutputDir + "/inference/mask/" + str(ts), mask)