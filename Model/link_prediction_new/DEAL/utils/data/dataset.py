import numpy as np
import torch
import networkx as nx
import glob
import re
from scipy.io import mmread
import scipy
import os
import random
import sys
import multiprocessing as mp

current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
from setting_param import ratio_test
from setting_param import ratio_valid

from setting_param import adj_shape
from setting_param import max_nnz_am
from setting_param import all_node_num
from setting_param import n_expanded
from setting_param import L
from setting_param import Model_link_prediction_appeared_InputDir as InputDir

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

def coo_scipy2coo_numpy(coo_scipy, max_nnz):
    # scipy_cooをvalue, indicesに分解してnumpy配列に変換
    # バッチ内のサイズを揃えるためにmax_nnzでゼロパディング
    coo_numpy = np.zeros((3, max_nnz))
    coo_numpy[:, :len(coo_scipy.data)] = np.vstack((coo_scipy.data, coo_scipy.row, coo_scipy.col))
    return coo_numpy

def in_out_generate(coo_numpy, n_node):
    coo_numpy_in = coo_numpy.copy()
    coo_numpy_out = np.zeros_like(coo_numpy)
    coo_numpy_out[0] = coo_numpy[0]
    coo_numpy_out[1] = coo_numpy[2] % n_node
    coo_numpy_out[2] = (coo_numpy[2] // n_node) * n_node + coo_numpy[1]
    return np.stack((coo_numpy_in, coo_numpy_out))

def get_A_in_per_batch(A, batch):
    A_in = A[batch][0]  # (3, max_nnz)
    nnz = int(A_in[0].sum().item())
    A_in = A_in[:, :nnz]  # (3, nnz)
    return A_in

def get_adj_per_t(A_in, t):
    col = A_in[2]
    idx = (t * all_node_num) <= col
    idx = idx * (col < ((t + 1) * all_node_num))
    A_t = A_in[:, idx]  # (3, nnz_per_t)
    A_t[2] = A_t[2] % all_node_num  # adjacency matrix per t
    return A_t

def get_cur_adj(A_t):
    cur_adj = {}
    cur_adj['vals'] = A_t[0]  # (nnz_per_t, )
    cur_adj['idx'] = A_t[1:].t().long()  # (nnz_per_t, 2)
    return cur_adj

def make_sparse_tensor(adj,tensor_type,torch_size):
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size*2)

    if tensor_type == 'float':
        test = torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
        return torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
    elif tensor_type == 'long':
        return torch.sparse.LongTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.long),
                                      tensor_size)
    else:
        raise NotImplementedError('only make floats or long sparse tensors')

def sparse_prepare_tensor(tensor,torch_size):
    tensor = make_sparse_tensor(tensor,
                                tensor_type = 'float',
                                torch_size = torch_size)
    return tensor

def get_A_last(A):
    """
    時系列の最後の隣接行列を取得
    """
    batch_adj_list = []
    for batch in range(A.shape[0]):
        A_in = get_A_in_per_batch(A, batch) # (3, nnz)
        A_t = get_adj_per_t(A_in, L-1) # (3, nnz_per_t)
        cur_adj = get_cur_adj(A_t)
        cur_adj = sparse_prepare_tensor(cur_adj, torch_size=[all_node_num])
        batch_adj_list.append(cur_adj)
    A_last = torch.stack(batch_adj_list, 0)
    return A_last

def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict

def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def all_pairs_shortest_path_length_parallel(graph,cutoff=None,num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes)<50:
        num_workers = int(num_workers/4)
    elif len(nodes)<400:
        num_workers = int(num_workers/2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range, args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict

def precompute_dist_data(edge_index, num_nodes, approximate=0):
    '''
    Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
    :return:
    '''
    graph = nx.Graph()
    edge_list = edge_index.tolist()
    graph.add_edges_from(edge_list)

    n = num_nodes
    dists_array = np.zeros((n, n))
    # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
    # dists_dict = {c[0]: c[1] for c in dists_dict}
    dists_dict = all_pairs_shortest_path_length_parallel(graph, cutoff=approximate if approximate > 0 else None)
    for i, node_i in enumerate(graph.nodes()):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(graph.nodes()):
            dist = shortest_dist.get(node_j, -1)
            if dist != -1:
                # dists_array[i, j] = 1 / (dist + 1)
                dists_array[node_i, node_j] = 1 / (dist + 1)
    return dists_array

class BADataset():
    def __init__(self, path, L, is_train, is_valid, is_test):
        # 入力ファイルのPATHのリストを取得
        attribute_paths = load_paths_from_dir(path + '/input')
        adjacency_paths = load_paths_from_dir(InputDir + '/input/adjacency')
        label_paths = load_paths_from_dir(path + '/label')
        mask_paths = load_paths_from_dir(path + '/mask')

        # split data
        n_samples = len(label_paths)
        all_idx = list(range(n_samples))
        dev_idx, test_idx = dev_test_split(all_idx, n_samples, ratio_test)
        train_idx, valid_idx = dev_test_split(dev_idx, n_samples, ratio_valid)

        if is_train:
            #target_idx = train_idx
            target_idx = [all_idx[-5]]
        elif is_valid:
            #target_idx = valid_idx
            target_idx = all_idx[-4:-2]
        elif is_test:
            #target_idx = test_idx
            target_idx = all_idx[-2:]
        else:
            target_idx = all_idx

        self.idx_list = target_idx
        self.attribute = [np.load(attribute_paths[idx]) for idx in target_idx]
        self.label = [mmread(label_paths[idx]).toarray() for idx in target_idx]
        self.mask = [mmread(mask_paths[idx]).toarray() for idx in target_idx]

        # adjacency: [n_sample * (2, 3, max_nnz_am)] → (n_sample, all_node_num, all_node_num) に変換
        self.adjacency = [in_out_generate(coo_scipy2coo_numpy(mmread(adjacency_paths[idx]), max_nnz_am), adj_shape[0])
                          for idx in target_idx]
        self.adjacency = get_A_last(torch.Tensor(self.adjacency))
        self.L = L

        self.data_arryas_link_ls = []
        self.dists_ls = []
        self.ind_train_A_ls = []
        self.ind_train_X_ls = []
        self.nodes_keep_ls = []
        self.A_ls = []
        self.X_ls = []

        for n in range(len(self.adjacency)):
            indic = self.adjacency[n]._indices().numpy().transpose(1, 0) # (nnz, 2) (有向グラフ：[0, 1]と[1, 0]が存在)
            indic_frozenset = set([frozenset({i, j}) for [i, j] in indic])
            graph = nx.Graph()
            graph.add_edges_from(indic.tolist())
            indic = np.array(list(graph.edges)) # (nnz/2, 2) (無向グラフ：[0, 1]と[1, 0]を区別しない)

            # valid_ones用をサンプリングする
            n_valid = indic.shape[0]//10
            idx_valid = random.sample(list(range(indic.shape[0])), n_valid)
            indic_valid_ones = indic[idx_valid]

            # valid_zeros用をサンプリングする
            indic_valid_zeros = set()
            while len(indic_valid_zeros) < len(indic_valid_ones):
                i = int(indic_valid_ones[random.sample(list(range(indic_valid_ones.shape[0])), 1), 0])
                j = int(indic[random.sample(list(range(indic.shape[0])), 1), 1])
                while i == j:
                    j = int(indic[random.sample(list(range(indic.shape[0])), 1), 1])
                if not frozenset([i, j]) in indic_frozenset:
                    indic_valid_zeros.add(frozenset([i, j]))
            indic_valid_zeros = np.array([list(edge) for edge in indic_valid_zeros])

            # train用をサンプリングする
            indic_train = np.delete(indic, obj=idx_valid, axis=0)

            # test用を整形する (オリジナルはones用とzeros用に分けているが自分の実験では全てを評価する)
            indic_test = []
            for i in range(all_node_num, all_node_num + n_expanded):
                for j in range(all_node_num + n_expanded):
                    indic_test.append([i, j])
            indic_test = np.array(indic_test)
            # test_ones用をサンプリングする (使わない)
            # indic_test_ones = torch.Tensor(self.label[n]).to_sparse()._indices().numpy().transpose(1, 0)  # (nnz, 2) (有向グラフ：[0, 1]と[1, 0]が存在)
            # graph = nx.Graph()
            # graph.add_edges_from(indic_test_ones.tolist())
            # indic_test_ones = np.array(list(graph.edges))  # (nnz/2, 2) (無向グラフ：[0, 1]と[1, 0]を区別しない)

            # data_arraysを作る (indic_train, indic_valid_ones, indic_valid_zeros, indic_test)
            data_arrays_link = [indic_train, indic_valid_ones, indic_valid_zeros, indic_test]
            #np.savez('data_arrays_link', indic_train, indic_valid_ones, indic_valid_zeros, indic_test)

            # distance matrixを計算する
            dists = precompute_dist_data(indic_train, all_node_num + n_expanded, approximate=-1)
            #np.save("dists-1", dists)

            # ind_train_A を作る
            ind_train_A = self.adjacency[n].to_dense().numpy()
            #ind_train_A = scipy.sparse.csc_matrix(self.adjacency[n].to_dense().numpy())
            #del_idx_ls = indic_valid_ones.transpose(1, 0).tolist()
            #ind_train_A[del_idx_ls[0], del_idx_ls[1]] = 0 # valのリンクを消す
            #ind_train_A[del_idx_ls[1], del_idx_ls[0]] = 0 # valのリンクを消す
            #scipy.sparse.save_npz('ind_train_A', ind_train_A)

            # ind_train_X を作る
            ind_train_X = self.attribute[n][:all_node_num]
            #ind_train_X = scipy.sparse.csc_matrix(self.attribute[n][:all_node_num])
            #scipy.sparse.save_npz('ind_train_X', ind_train_X)

            # nodes_keep
            nodes_keep = np.array(sorted(set(sum(indic_train.tolist(), []))))
            #np.save("nodes_keep", nodes_keep)

            # Aを作る
            A = np.zeros((all_node_num + n_expanded, all_node_num + n_expanded))
            A[:all_node_num, :all_node_num] = self.adjacency[n].to_dense().numpy()
            A = A + self.label[n]
            #A = scipy.sparse.csc_matrix(A + self.label[n])
            #scipy.sparse.save_npz('A_sp', A)

            # Xを作る
            X = self.attribute[n]
            #X = scipy.sparse.csc_matrix(self.attribute[n])
            #scipy.sparse.save_npz('X_sp', X)

            # チェック用
            indic_train_frozenset = set([frozenset({i, j}) for [i, j] in indic_train])
            indic_valid_ones_frozenset = set([frozenset({i, j}) for [i, j] in indic_valid_ones])
            indic_valid_zeros_frozenset = set([frozenset({i, j}) for [i, j] in indic_valid_zeros])
            indic_test_frozenset = set([frozenset({i, j}) for [i, j] in indic_test])
            assert indic_frozenset & indic_train_frozenset == indic_train_frozenset
            assert indic_frozenset & indic_valid_ones_frozenset == indic_valid_ones_frozenset
            assert indic_train_frozenset & indic_valid_ones_frozenset == set()
            assert indic_frozenset & indic_valid_zeros_frozenset == set()
            assert indic_frozenset & indic_test_frozenset == set()

            self.data_arryas_link_ls.append(data_arrays_link)
            self.dists_ls.append(dists)
            self.ind_train_A_ls.append(ind_train_A)
            self.ind_train_X_ls.append(ind_train_X)
            self.nodes_keep_ls.append(nodes_keep)
            self.A_ls.append(A)
            self.X_ls.append(X)

    def __getitem__(self, index):
        sample_idx = self.idx_list[index] + self.L
        data_arrays_link = self.data_arryas_link_ls[index]
        dists = self.dists_ls[index]
        ind_train_A = self.ind_train_A_ls[index]
        ind_train_X = self.ind_train_X_ls[index]
        nodes_keep = self.nodes_keep_ls[index]
        A = self.A_ls[index]
        X = self.X_ls[index]
        label = self.label[index]
        mask = self.mask[index]
        return sample_idx, data_arrays_link, dists, ind_train_A, ind_train_X, nodes_keep, A, X, label, mask

    def __len__(self):
        return len(self.idx_list)