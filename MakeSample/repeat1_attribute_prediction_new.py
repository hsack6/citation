import numpy as np
import os
import sys
import networkx as nx
from scipy.sparse import lil_matrix
from scipy.io import mmwrite, mmread

# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )

from setting_param import MakeSample_repeat1_attribute_prediction_new_InputDir as InputDir
from setting_param import MakeSample_repeat1_attribute_prediction_new_utilize_existing_attribute_OutputDir as OutputDir_0
from setting_param import MakeSample_repeat1_attribute_prediction_new_utilize_lost_OutputDir as OutputDir_1
from setting_param import MakeSample_repeat1_attribute_prediction_new_utilize_new_attribute_link_OutputDir as OutputDir_2
from setting_param import MakeSample_repeat1_attribute_prediction_new_utilize_disappeared_OutputDir as OutputDir_4
from setting_param import MakeSample_repeat1_attribute_prediction_new_utilize_appeared_OutputDir as OutputDir_8
OutputDir = {0:OutputDir_0, 1:OutputDir_1, 2:OutputDir_2, 4:OutputDir_4, 8:OutputDir_8}

from setting_param import L
from setting_param import attribute_dim
from setting_param import all_node_num
from setting_param import n_expanded

from repeat_utils.graph_prediction import link_prediction
from setting_param import best_methods
from setting_param import best_methods_attribute_new_OutputDir
from repeat_utils.get_predicted_new_attribute import get_predicted_new_attribute

for c_idx, Dir in OutputDir.items():
    os.makedirs(Dir, exist_ok=True)
    os.makedirs(Dir + "/input/", exist_ok=True)
    os.makedirs(Dir + "/input/node_attribute/", exist_ok=True)
    os.makedirs(Dir + "/input/adjacency", exist_ok=True)
    os.makedirs(Dir + "/label/", exist_ok=True)
    os.makedirs(Dir + "/mask/", exist_ok=True)

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

def TsSplit(ts, L):
    ts_train = [(ts+l) for l in range(L)]
    ts_test = ts_train[-1]+1
    ts_all = ts_train.copy()
    ts_all.extend([ts_test])
    return ts_train, ts_test, ts_all

def concat_train_valid_test(train_result, valid_result, test_result):
    result = []
    for train_r in train_result:
        result.append(train_r)
    for valid_r in valid_result:
        result.append(valid_r)
    for test_r in test_result:
        result.append(test_r)
    ts_result_dic = {}
    for t_idx, ts in enumerate(range(L, EXIST_TABLE.shape[1]-L)):
        ts_result_dic[ts] = result[t_idx]
    return ts_result_dic

# リンク予測の結果をそれぞれbest_methodのOutputDirから取得
train_result = link_prediction(best_methods["n_appeared"], best_methods["p_appeared"], best_methods["n_disappeared"], best_methods["p_disappeared"], best_methods["n_new"], best_methods["p_new"], best_methods["n_lost"],  best_methods["p_lost"], True, False, False)
valid_result = link_prediction(best_methods["n_appeared"], best_methods["p_appeared"], best_methods["n_disappeared"], best_methods["p_disappeared"], best_methods["n_new"], best_methods["p_new"], best_methods["n_lost"],  best_methods["p_lost"], False, True, False)
test_result = link_prediction(best_methods["n_appeared"], best_methods["p_appeared"], best_methods["n_disappeared"], best_methods["p_disappeared"], best_methods["n_new"], best_methods["p_new"], best_methods["n_lost"],  best_methods["p_lost"], False, False, True)
pred_adjacency_matrix = concat_train_valid_test(train_result, valid_result, test_result)

# new nodeのattributeの結果をそれぞれbest_methodのOutputDirから取得
train_new, train_teacher, train_new_num, train_teacher_num, train_teacher_idx, train_node_pair_list = get_predicted_new_attribute(best_methods_attribute_new_OutputDir, True, False, False)
valid_new, valid_teacher, valid_new_num, valid_teacher_num, valid_teacher_idx, valid_node_pair_list = get_predicted_new_attribute(best_methods_attribute_new_OutputDir, False, True, False)
test_new, test_teacher, test_new_num, test_teacher_num, test_teacher_idx, test_node_pair_list = get_predicted_new_attribute(best_methods_attribute_new_OutputDir, False, False, True)

pred_new_attribute_new = concat_train_valid_test(train_new, valid_new, test_new)
pred_new_attribute_teacher = concat_train_valid_test(train_teacher, valid_teacher, test_teacher)
pred_new_attribute_new_num = concat_train_valid_test(train_new_num, valid_new_num, test_new_num)
pred_new_attribute_teacher_num = concat_train_valid_test(train_teacher_num, valid_teacher_num, test_teacher_num)
pred_new_attribute_teacher_idx = concat_train_valid_test(train_teacher_idx, valid_teacher_idx, test_teacher_idx)
pred_new_attribute_node_pair_list = concat_train_valid_test(train_node_pair_list, valid_node_pair_list, test_node_pair_list)

pred_attribute = {}
for c_idx in range(16):
    if not c_idx in [2]:
        # 2 (new(属性＋リンク)予測のみ) だけ実験する
        continue
    pred_attribute[c_idx] = {}
    for ts in range(L, EXIST_TABLE.shape[1]-L):
        ts_train, ts_test, ts_all = TsSplit(ts, L)
        # existing nodeのattribute と new nodeのattribute を concatする
        # existing nodeのattributeの結果を取得 (変化しないので直前のものを持って来る)
        pred_attribute_e = NodeAttribute(ts_train[-1])
        # new node のattributeの結果を取得
        pred_attribute_n = pred_new_attribute_new[ts]
        if c_idx == 0:
            pred_attribute[c_idx][ts] = pred_attribute_e
        elif c_idx == 1:
            alive_nodes = set(np.unique(np.where(pred_adjacency_matrix[ts][c_idx] > 0)).tolist())
            lost_node_list = sorted(set(GetNodes(ts_train[-1], L, 'all')) - alive_nodes)
            pred_attribute_e[lost_node_list, :] = pred_attribute_e[lost_node_list, :] * 0
            pred_attribute[c_idx][ts] = pred_attribute_e
        elif c_idx == 2:
            pred_attribute[c_idx][ts] = np.concatenate([NodeAttribute(ts_train[-1]), pred_attribute_n], axis=0)
        else:
            pred_attribute[c_idx][ts] = NodeAttribute(ts_train[-1])
        #else:
        #    pred_attribute[c_idx][ts] = np.concatenate([pred_attribute_e, pred_attribute_n], axis=0) # (3948, 35) repeat1実験には無いが既知ノードと新規ノードの属性予測の結果を活用する場合


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
    teacher_num_ls = []
    teacher_idx_ls = []
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
    return np.array(new_ls), np.array(teacher_ls), np.array(new_num_ls), np.array(teacher_num_ls), np.array(
        teacher_idx_ls), np.array(node_pair_list_ls)


def load_npy_data(new_paths, teacher_paths, new_num_paths, teacher_num_paths, teacher_idx_paths,
                  node_pair_list_paths, all_idx, ts):
    idx = all_idx[ts - L]
    new = np.load(new_paths[idx])
    teacher = np.load(teacher_paths[idx])
    new_num = np.load(new_num_paths[idx])
    teacher_num = np.load(teacher_num_paths[idx])
    teacher_idx = np.load(teacher_idx_paths[idx])
    node_pair_list = np.load(node_pair_list_paths[idx])
    return new, teacher, new_num, teacher_num, teacher_idx, node_pair_list


# predicted_new_node_numの最大値を取得
from setting_param import Evaluation_prediction_num_of_node_new_LSTM_InputDir as predicted_num_InputDir

predicted_new_node_num_list = []
for ts in range(L, EXIST_TABLE.shape[1] - L):
    predicted_new_node_num = int(np.load(predicted_num_InputDir + '/output/pred' + str(ts) + '.npy')[0])
    predicted_new_node_num_list.append(predicted_new_node_num)
max_predicted_new_node_num = max(predicted_new_node_num_list)

# new_node_numの最大値を取得
new_node_num_list = []
for ts in range(L, EXIST_TABLE.shape[1] - L):
    ts_train, ts_test, ts_all = TsSplit(ts, L)
    new_node_num = len(GetNodes(ts_test, L, 'new'))
    new_node_num_list.append(new_node_num)
max_new_node_num = max(new_node_num_list)

n_expanded = max([max_predicted_new_node_num, max_new_node_num])

import glob
import re
from collections import defaultdict
from setting_param import Model_attribute_prediction_new_PROSER_selecter_OutputDir as PROSER_Out_InputDir
from setting_param import ratio_test
from setting_param import ratio_valid

def ordering_teacher(node_pair_list, new_num, teacher_num, teacher_idx, teacher, max_predicted_new_node_num, attribute_dim):
    node_pair_dic = {}
    for pair in node_pair_list:
        node_pair_dic[int(pair[0])] = int(pair[1])
    label = np.zeros((max_predicted_new_node_num, attribute_dim))
    for n in range(new_num):
        t_idx_abs = node_pair_dic[n]
        t_idx_rel = list(map(lambda x: int(x), teacher_idx[:teacher_num].tolist())).index(t_idx_abs)
        label[n] = teacher[t_idx_rel]
    return label

Model_Out_InputDir = PROSER_Out_InputDir
new_paths, teacher_paths, new_num_paths, teacher_num_paths, teacher_idx_paths, node_pair_list_paths = data_split(
    Model_Out_InputDir)
n_samples = len(new_paths)
all_idx = list(range(n_samples))
# dev_idx, test_idx = dev_test_split(all_idx, n_samples, ratio_test)
# train_idx, valid_idx = dev_test_split(dev_idx, n_samples, ratio_valid)
train_idx = all_idx[:-4]
valid_idx = all_idx[-4:-2]
test_idx = all_idx[-2:]

for c_idx in range(16):
    if not c_idx in [2]:
        # 2 (new(属性＋リンク)予測のみ) だけ実験する
        continue
    for ts in range(L, EXIST_TABLE.shape[1] - L):
        ts_train, ts_test, ts_all = TsSplit(ts, L)
        node_attribute = np.zeros((all_node_num + n_expanded, attribute_dim * L))
        npy_adjacency_matrix = np.zeros((all_node_num + n_expanded, (all_node_num + n_expanded) * L))

        # ノード属性, 隣接行列は t-L+2 から t までの観測値と t+1 の予測結果を使う
        for idx, ts_ in enumerate(ts_train[1:]):
            node_attribute[:all_node_num, attribute_dim * idx: attribute_dim * (idx + 1)] = NodeAttribute(ts_)
            npy_adjacency_matrix[:all_node_num,
            (all_node_num + n_expanded) * idx: (all_node_num + n_expanded) * idx + all_node_num] = get_adjacency_matrix(
                ts_, L, 'all')

        # ここでt+1の予測結果を加える
        node_attribute[:pred_attribute[c_idx][ts].shape[0], attribute_dim * (L - 1):] = pred_attribute[c_idx][ts]
        npy_adjacency_matrix[:, (all_node_num + n_expanded) * (L - 1):] = pred_adjacency_matrix[ts][c_idx]

        lil_adjacency_matrix = lil_matrix(npy_adjacency_matrix)
        lil_node_attribute = lil_matrix(node_attribute)

        mmwrite(OutputDir[c_idx] + "/input/node_attribute/" + str(ts), lil_node_attribute)
        mmwrite(OutputDir[c_idx] + "/input/adjacency/" + str(ts), lil_adjacency_matrix)

        new, teacher, new_num, teacher_num, teacher_idx, node_pair_list = load_npy_data(new_paths, teacher_paths,
                                                                                        new_num_paths,
                                                                                        teacher_num_paths,
                                                                                        teacher_idx_paths,
                                                                                        node_pair_list_paths, all_idx,
                                                                                        ts)

        # reference check
        assert sorted(GetNodes(ts_test, L, 'new')) == teacher_idx.tolist()[:teacher_num], 'reference error'
        predicted_new_node_num = int(np.load(predicted_num_InputDir + '/output/pred' + str(ts) + '.npy')[0])
        assert new_num == predicted_new_node_num, 'reference error'
        new_node_num = len(GetNodes(ts_test, L, 'new'))
        assert teacher_num == new_node_num, 'reference error'
        assert new.shape[0] == max_predicted_new_node_num, 'reference error'
        assert teacher.shape[0] == max_new_node_num, 'reference error'

        label = ordering_teacher(node_pair_list, new_num, teacher_num, teacher_idx, teacher, max_predicted_new_node_num, attribute_dim)
        mmwrite(OutputDir[c_idx] + "/label/" + str(ts), lil_matrix(label))

        mask = new_num
        mmwrite(OutputDir[c_idx] + "/mask/" + str(ts), lil_matrix(mask))