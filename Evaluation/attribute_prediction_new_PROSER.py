import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import copy
import os
import sys
from scipy.io import mmread

# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )

from setting_param import Evaluation_attribute_prediction_new_PROSER_InputDir as InputDir
from setting_param import Evaluation_attribute_prediction_new_PROSER_OutputDir as OutputDir
from setting_param import Evaluation_prediction_num_of_node_new_LSTM_InputDir as predicted_num_InputDir
from setting_param import L, ratio_test, ratio_valid
from statistics import mean, median

os.makedirs(OutputDir, exist_ok=True)

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

def NodeAttribute(ts):
    return np.load(InputDir + '/node_attribute' + str(ts) + '.npy')

def TsSplit(ts, L):
    ts_train = [(ts+l) for l in range(L)]
    ts_test = ts_train[-1]+1
    ts_all = ts_train.copy()
    ts_all.extend([ts_test])
    return ts_train, ts_test, ts_all

def Matching(n_list_tup, t_list_tup, n, t):
    pair_list = []
    decided_n = set()
    decided_t = set()
    while True:
        for idx in range(len(n)):
            matched_n, matched_t = n[idx], t[idx]
            if matched_n in decided_n or matched_t in decided_t:
                continue
            pair_list.append((n_list_tup[matched_n][0], t_list_tup[matched_t][0]))
            decided_n.add(matched_n)
            decided_t.add(matched_t)
            if len(decided_n) == len(n_list_tup):
                break
            if len(decided_t) == len(t_list_tup):
                decided_t = set()
                break
        if len(decided_n) == len(n_list_tup):
            break
    return pair_list

def BipartiteMatching(new_vec_dic, teacher_vec_dic):
    eps = 0.000001 # zero-division errorの回避用
    # sort済みのタプルのリスト[(node_id, vector)]
    n_list_tup = sorted(new_vec_dic.items(), key=lambda x: x[0])
    t_list_tup = sorted(teacher_vec_dic.items(), key=lambda x: x[0])
    # 二部グラフ全てのsimilarityを計算
    N = np.array([n_v for n, n_v in n_list_tup])
    T = np.array([t_v for t, t_v in t_list_tup])
    normN = np.sqrt(np.sum(N * N, axis=1)) + eps
    normT = np.sqrt(np.sum(T * T, axis=1)) + eps
    similarity_matrix = np.dot(N / normN.reshape(-1, 1), (T / normT.reshape(-1, 1)).T)
    # similarityのsort
    n, t = np.unravel_index(np.argsort(-similarity_matrix.reshape(-1)), similarity_matrix.shape)
    # Greedy Matching
    node_pair_list = Matching(copy.copy(n_list_tup), copy.copy(t_list_tup), n.tolist(), t.tolist())
    return node_pair_list, similarity_matrix

def dev_test_split(all_idx, n_samples, ratio_test):
    n_test = int(n_samples * ratio_test)
    return all_idx[:-n_test], all_idx[-n_test:]

def train_valid_split(dev_idx, n_samples, ratio_valid):
    n_valid = int(n_samples * ratio_valid)
    return dev_idx[:-n_valid], dev_idx[-n_valid:]

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

all_idx = [ts for ts in range(L, EXIST_TABLE.shape[1] - L)]
train_idx = all_idx[-18:-4]
valid_idx = all_idx[-4:-2]
test_idx = all_idx[-2:]

new = []
teacher = []
new_num = []
teacher_num = []
teacher_idx = []
labels = []

for ts in range(L, EXIST_TABLE.shape[1] - L):
    # valid 区間だけを見る
    if ts not in valid_idx:
        continue

    ts_train, ts_test, ts_all = TsSplit(ts, L)

    node_attribute_new = NodeAttribute(ts_train[-1])[sorted(GetNodes(ts_train[-1], L, 'new'))]  # (921, 300)
    predicted_new_node_num = int(np.load(predicted_num_InputDir + '/output/pred' + str(ts) + '.npy')[0])
    new_ = np.zeros((max_predicted_new_node_num, node_attribute_new.shape[1]))
    input_matrix = np.empty((0, node_attribute_new.shape[1]), float)
    for _ in range(predicted_new_node_num // node_attribute_new.shape[0]):
        input_matrix = np.append(input_matrix, node_attribute_new, axis=0)
    input_matrix = np.append(input_matrix, node_attribute_new[:predicted_new_node_num % node_attribute_new.shape[0]], axis=0)
    new_[:input_matrix.shape[0]] = input_matrix
    new.append(new_)

    new_num.append(input_matrix.shape[0])

    node_attribute_teacher = NodeAttribute(ts_test)[sorted(GetNodes(ts_test, L, 'new'))]  # (1023, 300)
    teacher_ = np.zeros((max_new_node_num, node_attribute_teacher.shape[1]))
    teacher_[:node_attribute_teacher.shape[0]] = node_attribute_teacher
    teacher.append(teacher_)

    teacher_num.append(node_attribute_teacher.shape[0])

    teacher_idx_ = np.zeros(max_new_node_num)
    teacher_idx_[:len(sorted(GetNodes(ts_test, L, 'new')))] = sorted(GetNodes(ts_test, L, 'new'))
    teacher_idx.append(teacher_idx_)

    # sim_matrix = cosine_similarity(node_attribute_new, node_attribute_teacher)  # (921, 1023)
    # sim_matrix_min = sim_matrix.min(axis=1)
    # sim_matrix_25 = np.percentile(sim_matrix, 25, axis=1)
    # sim_matrix_50 = np.percentile(sim_matrix, 50, axis=1)
    # sim_matrix_75 = np.percentile(sim_matrix, 75, axis=1)
    # sim_matrix_max = sim_matrix.max(axis=1)
    # sim_matrix_stats = np.array([sim_matrix_min, sim_matrix_25, sim_matrix_50, sim_matrix_75, sim_matrix_max]).transpose((1, 0))  # (921, 5)
    # label_matrix = np.zeros((max_new_node_num, 5))
    # label_matrix[:sim_matrix_stats.shape[0]] = sim_matrix_stats
    # labels.append(label_matrix[:, target_idx])

from setting_param import Model_attribute_prediction_new_PROSER_FNN_OutputDir as MPFO
labels = []
for ts in range(L, EXIST_TABLE.shape[1] - L):
    # valid 区間だけを見る
    if ts not in valid_idx:
        continue
    labels.append(np.load(MPFO + "/output/output" + str(ts) + '.npy')[:, 0])

l = []
for i in range(len(labels)):
    for j in range(new_num[i]):
        l.append(labels[i][j])
th_ls = []
for i in range(21):
    th_ls.append(np.percentile(l, i))

gain_mean_ls = []
gain_min_ls = []
gain_median_ls = []
gain_max_ls = []

for th in th_ls:
    # 閾値で追加するかどうかの選択を行うver
    new = []
    for ts in range(L, EXIST_TABLE.shape[1] - L):
        # valid 区間だけを見る
        if ts not in valid_idx:
            continue

        ts_train, ts_test, ts_all = TsSplit(ts, L)

        node_attribute_new = NodeAttribute(ts_train[-1])[sorted(GetNodes(ts_train[-1], L, 'new'))]  # (921, 300)
        predicted_new_node_num = int(np.load(predicted_num_InputDir + '/output/pred' + str(ts) + '.npy')[0])
        new_ = np.zeros((max_predicted_new_node_num, node_attribute_new.shape[1]))
        input_matrix = np.zeros((predicted_new_node_num, node_attribute_new.shape[1]))
        insert_idx = 0

        explore_idx = np.array([j for j in range(node_attribute_new.shape[0])])[labels[ts - valid_idx[0]][:node_attribute_new.shape[0]] >= th]
        while True:
            for i in explore_idx:
                input_matrix[insert_idx] = node_attribute_new[i]
                insert_idx += 1
                if insert_idx == predicted_new_node_num:
                    break
            if insert_idx == predicted_new_node_num:
                break
        new_[:input_matrix.shape[0]] = input_matrix
        new.append(new_)

    gain_mean = 0
    gain_min = 0
    gain_median = 0
    gain_max = 0

    for batch in range(len(new)):

        new_vec_dic = {i: new[batch][i].tolist() for i in range(new_num[batch])}
        teacher_vec_dic = {i: teacher[batch][i].tolist() for i in range(teacher_num[batch])}
        node_pair_list, similarity_matrix = BipartiteMatching(new_vec_dic, teacher_vec_dic)

        score = []
        for i in range(new_num[batch]):
            score.append(similarity_matrix[node_pair_list[i]])
        gain_mean += mean(score)
        gain_min += min(score)
        gain_median += median(score)
        gain_max += max(score)

        node_pair_list = [(i, teacher_idx[batch][j]) for (i, j) in node_pair_list]

    gain_mean_ls.append(gain_mean / len(new))
    gain_min_ls.append(gain_min / len(new))
    gain_median_ls.append(gain_median / len(new))
    gain_max_ls.append(gain_max / len(new))

    print("Percentile:", len(gain_mean_ls)-1)
    print(gain_mean / len(new))
    print(gain_min / len(new))
    print(gain_median / len(new))
    print(gain_max / len(new))

result_dic = {"Threshold":th_ls, "Gain(mean)":gain_mean_ls, "Gain(min)":gain_min_ls, "Gain(median)":gain_median_ls, "Gain(max)":gain_max_ls}
df = pd.DataFrame(result_dic)
df.to_csv(OutputDir + '/result_oracle_valid.csv')

plt.figure()
plt.plot(th_ls, gain_mean_ls, label="mean")
plt.xlabel('Threshold')
plt.ylabel('Gain')
plt.title('mean')
plt.savefig(OutputDir + '/result_oracle_valid_mean.pdf')

plt.figure()
plt.plot(th_ls, gain_min_ls, label="min")
plt.xlabel('Threshold')
plt.ylabel('Gain')
plt.title('min')
plt.savefig(OutputDir + '/result_oracle_valid_min.pdf')

plt.figure()
plt.plot(th_ls, gain_median_ls, label="median")
plt.xlabel('Threshold')
plt.ylabel('Gain')
plt.title('median')
plt.savefig(OutputDir + '/result_oracle_valid_median.pdf')

plt.figure()
plt.plot(th_ls, gain_max_ls, label="max")
plt.xlabel('Threshold')
plt.ylabel('Gain')
plt.title('max')
plt.savefig(OutputDir + '/result_oracle_valid_max.pdf')


new = []
teacher = []
new_num = []
teacher_num = []
teacher_idx = []
labels = []

for ts in range(L, EXIST_TABLE.shape[1] - L):
    # test 区間だけを見る
    if ts not in test_idx:
        continue

    ts_train, ts_test, ts_all = TsSplit(ts, L)

    node_attribute_new = NodeAttribute(ts_train[-1])[sorted(GetNodes(ts_train[-1], L, 'new'))]  # (921, 300)
    predicted_new_node_num = int(np.load(predicted_num_InputDir + '/output/pred' + str(ts) + '.npy')[0])
    new_ = np.zeros((max_predicted_new_node_num, node_attribute_new.shape[1]))
    input_matrix = np.empty((0, node_attribute_new.shape[1]), float)
    for _ in range(predicted_new_node_num // node_attribute_new.shape[0]):
        input_matrix = np.append(input_matrix, node_attribute_new, axis=0)
    input_matrix = np.append(input_matrix, node_attribute_new[:predicted_new_node_num % node_attribute_new.shape[0]], axis=0)
    new_[:input_matrix.shape[0]] = input_matrix
    new.append(new_)

    new_num.append(input_matrix.shape[0])

    node_attribute_teacher = NodeAttribute(ts_test)[sorted(GetNodes(ts_test, L, 'new'))]  # (1023, 300)
    teacher_ = np.zeros((max_new_node_num, node_attribute_teacher.shape[1]))
    teacher_[:node_attribute_teacher.shape[0]] = node_attribute_teacher
    teacher.append(teacher_)

    teacher_num.append(node_attribute_teacher.shape[0])

    teacher_idx_ = np.zeros(max_new_node_num)
    teacher_idx_[:len(sorted(GetNodes(ts_test, L, 'new')))] = sorted(GetNodes(ts_test, L, 'new'))
    teacher_idx.append(teacher_idx_)

    # sim_matrix = cosine_similarity(node_attribute_new, node_attribute_teacher)  # (921, 1023)
    # sim_matrix_min = sim_matrix.min(axis=1)
    # sim_matrix_25 = np.percentile(sim_matrix, 25, axis=1)
    # sim_matrix_50 = np.percentile(sim_matrix, 50, axis=1)
    # sim_matrix_75 = np.percentile(sim_matrix, 75, axis=1)
    # sim_matrix_max = sim_matrix.max(axis=1)
    # sim_matrix_stats = np.array([sim_matrix_min, sim_matrix_25, sim_matrix_50, sim_matrix_75, sim_matrix_max]).transpose((1, 0))  # (921, 5)
    # label_matrix = np.zeros((max_new_node_num, 5))
    # label_matrix[:sim_matrix_stats.shape[0]] = sim_matrix_stats
    # labels.append(label_matrix[:, target_idx])

from setting_param import Model_attribute_prediction_new_PROSER_FNN_OutputDir as MPFO
labels = []
for ts in range(L, EXIST_TABLE.shape[1] - L):
    # test 区間だけを見る
    if ts not in test_idx:
        continue
    labels.append(np.load(MPFO + "/output/output" + str(ts) + '.npy')[:, 0])

l = []
for i in range(len(labels)):
    for j in range(new_num[i]):
        l.append(labels[i][j])
th_ls = []
for i in range(21):
    th_ls.append(np.percentile(l, i))

gain_mean_ls = []
gain_min_ls = []
gain_median_ls = []
gain_max_ls = []

for th in th_ls:
    # 閾値で追加するかどうかの選択を行うver
    new = []
    for ts in range(L, EXIST_TABLE.shape[1] - L):
        # test 区間だけを見る
        if ts not in test_idx:
            continue

        ts_train, ts_test, ts_all = TsSplit(ts, L)

        node_attribute_new = NodeAttribute(ts_train[-1])[sorted(GetNodes(ts_train[-1], L, 'new'))]  # (921, 300)
        predicted_new_node_num = int(np.load(predicted_num_InputDir + '/output/pred' + str(ts) + '.npy')[0])
        new_ = np.zeros((max_predicted_new_node_num, node_attribute_new.shape[1]))
        input_matrix = np.zeros((predicted_new_node_num, node_attribute_new.shape[1]))
        insert_idx = 0

        explore_idx = np.array([j for j in range(node_attribute_new.shape[0])])[labels[ts - test_idx[0]][:node_attribute_new.shape[0]] >= th]
        while True:
            for i in explore_idx:
                input_matrix[insert_idx] = node_attribute_new[i]
                insert_idx += 1
                if insert_idx == predicted_new_node_num:
                    break
            if insert_idx == predicted_new_node_num:
                break
        new_[:input_matrix.shape[0]] = input_matrix
        new.append(new_)

    gain_mean = 0
    gain_min = 0
    gain_median = 0
    gain_max = 0

    for batch in range(len(new)):

        new_vec_dic = {i: new[batch][i].tolist() for i in range(new_num[batch])}
        teacher_vec_dic = {i: teacher[batch][i].tolist() for i in range(teacher_num[batch])}
        node_pair_list, similarity_matrix = BipartiteMatching(new_vec_dic, teacher_vec_dic)

        score = []
        for i in range(new_num[batch]):
            score.append(similarity_matrix[node_pair_list[i]])
        gain_mean += mean(score)
        gain_min += min(score)
        gain_median += median(score)
        gain_max += max(score)

        node_pair_list = [(i, teacher_idx[batch][j]) for (i, j) in node_pair_list]

    gain_mean_ls.append(gain_mean / len(new))
    gain_min_ls.append(gain_min / len(new))
    gain_median_ls.append(gain_median / len(new))
    gain_max_ls.append(gain_max / len(new))

    print("Percentile:", len(gain_mean_ls)-1)
    print(gain_mean / len(new))
    print(gain_min / len(new))
    print(gain_median / len(new))
    print(gain_max / len(new))

result_dic = {"Threshold":th_ls, "Gain(mean)":gain_mean_ls, "Gain(min)":gain_min_ls, "Gain(median)":gain_median_ls, "Gain(max)":gain_max_ls}
df = pd.DataFrame(result_dic)
df.to_csv(OutputDir + '/result_oracle_test.csv')

plt.figure()
plt.plot(th_ls, gain_mean_ls, label="mean")
plt.xlabel('Threshold')
plt.ylabel('Gain')
plt.title('mean')
plt.savefig(OutputDir + '/result_oracle_test_mean.pdf')

plt.figure()
plt.plot(th_ls, gain_min_ls, label="min")
plt.xlabel('Threshold')
plt.ylabel('Gain')
plt.title('min')
plt.savefig(OutputDir + '/result_oracle_test_min.pdf')

plt.figure()
plt.plot(th_ls, gain_median_ls, label="median")
plt.xlabel('Threshold')
plt.ylabel('Gain')
plt.title('median')
plt.savefig(OutputDir + '/result_oracle_test_median.pdf')

plt.figure()
plt.plot(th_ls, gain_max_ls, label="max")
plt.xlabel('Threshold')
plt.ylabel('Gain')
plt.title('max')
plt.savefig(OutputDir + '/result_oracle_test_max.pdf')