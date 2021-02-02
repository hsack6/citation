import numpy as np
from scipy.io import mmread
import seaborn as sns
import networkx as nx
import glob
import re
import os
import sys
sns.set(style='darkgrid')
sns.set_style(style='whitegrid')

# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )

from setting_param import MakeSample_link_prediction_appeared_InputDir
EXIST_TABLE = np.load(MakeSample_link_prediction_appeared_InputDir + '/exist_table.npy')

from setting_param import L
from setting_param import ratio_test
from setting_param import ratio_valid
from setting_param import all_node_num
from setting_param import n_expanded

from repeat2_utils.graph_prediction_IO import get_new_InputDirs # new link

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


def load_output_data(true_paths, pred_paths, mask_paths, target_idx):
    y_true = []
    y_pred = []
    y_mask = []
    for idx in target_idx:
        true = mmread(true_paths[idx]).toarray()
        pred = mmread(pred_paths[idx]).toarray()
        mask = mmread(mask_paths[idx]).toarray()
        y_true.append(true.tolist())
        y_pred.append(pred.tolist())
        y_mask.append(mask.tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mask = np.array(y_mask)
    return y_true, y_pred, y_mask


def get_true_pred_mask(InputDir, is_train, is_valid, is_test):
    true_paths, pred_paths, mask_paths = true_pred_mask_split(InputDir)
    n_samples = len(true_paths)
    all_idx = list(range(n_samples))
    # dev_idx, test_idx = dev_test_split(all_idx, n_samples, ratio_test)
    # train_idx, valid_idx = dev_test_split(dev_idx, n_samples, ratio_valid)
    train_idx = all_idx[:-4]
    valid_idx = all_idx[-4:-2]
    test_idx = all_idx[-2:]

    if is_train:
        target_idx = train_idx
    elif is_valid:
        target_idx = valid_idx
    elif is_test:
        target_idx = test_idx

    y_true, y_pred, y_mask = load_output_data(true_paths, pred_paths, mask_paths, target_idx)
    return y_true, y_pred, y_mask


def get_predicted_num_list(predicted_num_list, is_train, is_valid, is_test):
    n_samples = len(predicted_num_list)
    all_idx = list(range(n_samples))
    # dev_idx, test_idx = dev_test_split(all_idx, n_samples, ratio_test)
    # train_idx, valid_idx = dev_test_split(dev_idx, n_samples, ratio_valid)
    train_idx = all_idx[:-4]
    valid_idx = all_idx[-4:-2]
    test_idx = all_idx[-2:]

    if is_train:
        target_idx = train_idx
    elif is_valid:
        target_idx = valid_idx
    elif is_test:
        target_idx = test_idx

    return np.array(predicted_num_list)[target_idx]


def sort_pred(true_, pred_, coordinate_):
    t, p, c = true_, pred_, coordinate_
    c_idx = list(range(len(c)))
    tmp_for_sort = list(zip(t.tolist(), p.tolist(), c_idx))
    tmp_for_sort = list(zip(*sorted(tmp_for_sort, key=lambda x: -x[1])))
    t = np.array(tmp_for_sort[0])
    p = np.array(tmp_for_sort[1])
    c = c[list(tmp_for_sort[2])]
    return (t, p, c)


def edge_coordinate_subset_pred(c, p, n):
    c = np.array(c, dtype=int).tolist()
    p = p.tolist()
    targets = set()
    while len(targets) < n:
        if p[0] < 0.0001:
            break
        targets.add(frozenset(c.pop(0)))
        p.pop(0)
    return targets


def edge_coordinate_subset_true(c, t):
    c = np.array(c, dtype=int).tolist()
    targets = set()
    for i, edge in enumerate(np.array(t, dtype=int).tolist()):
        if edge == 1:
            targets.add(frozenset(c[i]))
    return targets


def calc_edge_recall_precision(y_true, y_pred, y_mask, predicted_edge_num_list, n_node):
    """
    coordinate                      : (n_node, n_node, 2)
    true                            : (n_node, n_node)
    pred                            : (n_node, n_node)
    mask                            : (n_node, n_node)
    true_                           : (n_candidate)
    pred_                           : (n_candidate)
    coordinate_                     : (n_candidate, 2)
    """
    # 座標配列を生成
    coordinate = np.zeros((n_node, n_node, 2))
    for row in range(coordinate.shape[0]):
        for column in range(coordinate.shape[1]):
            coordinate[row][column][0] = row
            coordinate[row][column][1] = column

    n_pred = 0
    n_true = 0
    n_true_and_pred = 0
    pred_set_list = []
    true_set_list = []

    for sample_idx in range(y_true.shape[0]):
        # true, predを取得
        true = y_true[sample_idx]
        pred = y_pred[sample_idx]
        # 予測された数を取得
        n = predicted_edge_num_list[sample_idx]
        # maskをかける
        mask = y_mask[sample_idx]
        true_ = true[0 < mask]
        pred_ = pred[0 < mask]
        coordinate_ = coordinate[0 < mask]
        # predの高い順にtrue_, pred_, coordinate_をソート
        true_, pred_, coordinate_ = sort_pred(true_, pred_, coordinate_)
        # coordinate_の上から順に予測された数のリンクを追加（※対称座標は除去=無向グラフ）、これが予測されたリンク集合となる。
        pred_set = edge_coordinate_subset_pred(coordinate_, pred_, n)
        # 正解のリンク集合を取得
        true_set = edge_coordinate_subset_true(coordinate_, true_)
        # 正解集合と予測集合、および積集合の要素をカウント
        n_pred += len(pred_set)
        n_true += len(true_set)
        n_true_and_pred += len(true_set & pred_set)
        # 正解集合と予測集合を保持
        pred_set_list.append(pred_set)
        true_set_list.append(true_set)
    recall = -1 if n_true == 0 else n_true_and_pred / n_true
    precision = -1 if n_pred == 0 else n_true_and_pred / n_pred
    return recall, precision, pred_set_list, true_set_list


def node_coordinate_subset_pred(c, p, n):
    c = np.array(c, dtype=int).tolist()
    p = p.tolist()
    targets = set()
    while len(targets) < n:
        if p[0] < 0.0001:
            break
        targets.add(c.pop(0))
        p.pop(0)
    return targets


def node_coordinate_subset_true(c, t):
    c = np.array(c, dtype=int).tolist()
    targets = set()
    for i, node in enumerate(np.array(t, dtype=int).tolist()):
        if node == 1:
            targets.add(c[i])
    return targets


def calc_node_recall_precision(y_true, y_pred, y_mask, predicted_node_num_list, n_node):
    """
    coordinate      : (n_node, 1, 1)
    true            : (n_node, 1)
    pred            : (n_node, 1)
    mask            : (n_node, 1)
    true_           : (n_candidate)
    pred_           : (n_candidate)
    coordinate_     : (n_candidate)
    """
    # 座標配列を生成
    coordinate = np.zeros((n_node, 1))
    for row in range(coordinate.shape[0]):
        coordinate[row][0] = row

    n_pred = 0
    n_true = 0
    n_true_and_pred = 0
    pred_set_list = []
    true_set_list = []

    for sample_idx in range(y_true.shape[0]):
        # true, predを取得
        true = y_true[sample_idx]
        pred = y_pred[sample_idx]
        # 予測された数を取得
        n = predicted_node_num_list[sample_idx]
        # maskをかける
        mask = y_mask[sample_idx]
        true_ = true[0 < mask]
        pred_ = pred[0 < mask]
        coordinate_ = coordinate[0 < mask]
        # predの高い順にtrue_, pred_, coordinate_をソート
        true_, pred_, coordinate_ = sort_pred(true_, pred_, coordinate_)
        # coordinate_の上から順に予測された数のノードを追加
        pred_set = node_coordinate_subset_pred(coordinate_, pred_, n)
        # 正解のリンク集合を取得
        true_set = node_coordinate_subset_true(coordinate_, true_)
        # 正解集合と予測集合、および積集合の要素をカウント
        n_pred += len(pred_set)
        n_true += len(true_set)
        n_true_and_pred += len(true_set & pred_set)
        # 正解集合と予測集合を保持
        pred_set_list.append(pred_set)
        true_set_list.append(true_set)
    recall = -1 if n_true == 0 else n_true_and_pred / n_true
    precision = -1 if n_pred == 0 else n_true_and_pred / n_pred
    return recall, precision, pred_set_list, true_set_list


def get_ts_list(InputDir, is_train, is_valid, is_test):
    true_paths, pred_paths, mask_paths = true_pred_mask_split(InputDir)
    n_samples = len(true_paths)
    all_idx = list(range(n_samples))
    # dev_idx, test_idx = dev_test_split(all_idx, n_samples, ratio_test)
    # train_idx, valid_idx = dev_test_split(dev_idx, n_samples, ratio_valid)
    train_idx = all_idx[:-4]
    valid_idx = all_idx[-4:-2]
    test_idx = all_idx[-2:]
    if is_train:
        target_idx = train_idx
    elif is_valid:
        target_idx = valid_idx
    elif is_test:
        target_idx = test_idx

    ts_list = list(map(lambda x: x + L, target_idx))
    return ts_list


def TsSplit(ts, L):
    ts_train = [(ts + l) for l in range(L)]
    ts_test = ts_train[-1] + 1
    ts_all = ts_train.copy()
    ts_all.extend([ts_test])
    return ts_train, ts_test, ts_all


def get_component_result(component_type, probability_InputDir, num_InputDir, node_num, is_train, is_valid, is_test):
    predicted_num_list = []
    for ts in range(L, EXIST_TABLE.shape[1] - L):
        predicted_num = int(np.load(num_InputDir + '/output/pred' + str(ts) + '.npy'))
        predicted_num_list.append(predicted_num)

    y_true, y_pred, y_mask = get_true_pred_mask(probability_InputDir, is_train, is_valid, is_test)
    y_predicted_num = get_predicted_num_list(predicted_num_list, is_train, is_valid, is_test)
    if component_type == "node":
        recall, precision, pred_set_list, true_set_list = calc_node_recall_precision(y_true, y_pred, y_mask, y_predicted_num, node_num)
    elif component_type == "edge":
        recall, precision, pred_set_list, true_set_list = calc_edge_recall_precision(y_true, y_pred, y_mask, y_predicted_num, node_num)
    if precision + recall < 0.000001:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    print(num_InputDir)
    print(probability_InputDir)
    print("recall: ", recall, "precision: ", precision, "f1: ", f1_score)
    return pred_set_list, true_set_list, recall, precision, f1_score


def link_prediction(n_appeared, p_appeared, n_disappeared, p_disappeared, n_new, p_new, n_lost, p_lost, is_train, is_valid, is_test):

    probability_new_InputDir, num_new_InputDir = get_new_InputDirs(p_new, n_new)
    new_edge_pred_set_list, new_edge_true_set_list, recall_new_edge, precision_new_edge, f1_score_new_edge = get_component_result("edge", probability_new_InputDir, num_new_InputDir, all_node_num + n_expanded, is_train, is_valid, is_test)

    # 総合結果を計算
    # 「tのlink集合 」 + 「appeared (link) 集合」+ 「new (link) 集合」- 「disappeared (link) 集合」- 「lost (link) 集合」
    ts_list = get_ts_list(probability_new_InputDir, is_train, is_valid, is_test)
    ts_c_pred_A = []
    for i, ts in enumerate(ts_list):
        ts_train, ts_test, ts_all = TsSplit(ts, L)
        t_edge_set = set()
        for edge in nx.from_numpy_matrix(mmread(MakeSample_link_prediction_appeared_InputDir + '/adjacency' + str(ts_train[-1])).toarray()).edges:
            t_edge_set.add(frozenset({edge[0], edge[1]}))

        new_edge_pred_set = new_edge_pred_set_list[i]
        new_edge_true_set = new_edge_true_set_list[i]
        assert len(t_edge_set & new_edge_true_set) == 0, "tのlink集合とnew(link)集合は被らない"
        assert len(t_edge_set & new_edge_pred_set) == 0, "tのlink集合とnew(link)集合は被らない"

        pred_set = [set() for _ in range(16)]

        # appeared : disappeared : new : lost
        # 何もしない場合 0000
        pred_set[0] = t_edge_set
        # lostのみをbest methodにした時 0001
        pred_set[1] = set()
        # newのみをbest methodにした時 0010
        pred_set[2] = t_edge_set | new_edge_pred_set
        # lostとnewのみをbest methodにした時 0011
        pred_set[3] = set()
        # disappearedのみをbest methodにした時 0100
        pred_set[4] = set()
        # disappearedとlostをbest methodにした時 0101
        pred_set[5] = set()
        # disappearedとnewをbest methodにした時 0110
        pred_set[6] = set()
        # disappearedとnewとlostをbest methodにした時 0111
        pred_set[7] = set()
        # appearedのみをbest methodにした時 1000
        pred_set[8] = set()
        # appearedとlostをbest methodにした時 1001
        pred_set[9] = set()
        # appearedとnewをbest methodにした時 1010
        pred_set[10] = set()
        # appearedとnewとlostをbest methodにした時 1011
        pred_set[11] = set()
        # appearedとdisappearedのみをbest methodにした時 1100
        pred_set[12] = set()
        # appearedとdisappearedとlostのみをbest methodにした時 1101
        pred_set[13] = set()
        # appearedとdisappearedとnewのみをbest methodにした時 1110
        pred_set[14] = set()
        # appearedとdisappearedとnewとlostをbest methodにした時 1111
        pred_set[15] = set()

        pred_A_list = []
        for c_idx in range(16):
            if c_idx == 2:
                pred_G = nx.Graph()
                pred_G.add_edges_from([tuple(froset) for froset in pred_set[c_idx]])
                pred_A = np.array(nx.to_numpy_matrix(pred_G, nodelist=[node for node in range(all_node_num + n_expanded)]))
                pred_A_list.append(pred_A)
            else:
                pred_A_list.append(np.zeros(1))
        ts_c_pred_A.append(pred_A_list)

    return np.array(ts_c_pred_A)