import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import re
import os
import copy
import sys
from statistics import mean

sns.set(style='darkgrid')
sns.set_style(style='whitegrid')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )
from setting_param import ratio_test
from setting_param import ratio_valid
from setting_param import MakeSample_attribute_prediction_new_InputDir

EXIST_TABLE = np.load(MakeSample_attribute_prediction_new_InputDir + '/exist_table.npy')

from setting_param import Evaluation_attribute_prediction_new_Baseline_InputDir as Baseline_InputDir
from setting_param import Evaluation_attribute_prediction_new_FNN_InputDir as FNN_InputDir
from setting_param import Evaluation_attribute_prediction_new_DeepMatchMax_InputDir as DeepMatchMax_InputDir

from setting_param import Evaluation_attribute_prediction_new_OutputDir as OutputDir

InputDirs = [Baseline_InputDir, FNN_InputDir, DeepMatchMax_InputDir]
methods = ['Baseline', 'FNN', 'DeepMatchMax']
os.makedirs(OutputDir, exist_ok=True)

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
    return np.array(new_ls), np.array(teacher_ls), np.array(new_num_ls), np.array(teacher_num_ls)

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


def calc_metrics(new_paths, teacher_paths, new_num_paths, teacher_num_paths, target_idx):
    gain_mean = 0
    gain_min = 0
    gain_max = 0
    new_ls = []
    teacher_ls = []
    for idx in target_idx:
        new = np.load(new_paths[idx])
        teacher = np.load(teacher_paths[idx])
        new_num = np.load(new_num_paths[idx])
        teacher_num = np.load(teacher_num_paths[idx])

        new_vec_dic = {i: new[i].tolist() for i in range(new_num)}
        teacher_vec_dic = {i: teacher[i].tolist() for i in range(teacher_num)}
        node_pair_list, similarity_matrix = BipartiteMatching(new_vec_dic, teacher_vec_dic)

        transformed_new = new[:new_num]
        transformed_teacher = np.zeros((new_num, teacher.shape[1]))
        for (n_idx, t_idx) in node_pair_list:
            transformed_teacher[n_idx] = teacher[t_idx]
        new_ls.append(transformed_new)
        teacher_ls.append(transformed_teacher)

        score = []
        for i in range(new_num):
            score.append(similarity_matrix[node_pair_list[i]])
        gain_mean += mean(score)
        gain_min += min(score)
        gain_max += max(score)

    gain_mean /= len(target_idx)
    gain_min /= len(target_idx)
    gain_max /= len(target_idx)

    return gain_mean, gain_min, gain_max

def get_performance(InputDir, is_train, is_valid, is_test):
    new_paths, teacher_paths, new_num_paths, teacher_num_paths = data_split(InputDir)
    n_samples = len(new_paths)
    all_idx = list(range(n_samples))
    dev_idx, test_idx = dev_test_split(all_idx, n_samples, ratio_test)
    train_idx, valid_idx = dev_test_split(dev_idx, n_samples, ratio_valid)

    if is_train:
        target_idx = train_idx
    elif is_valid:
        target_idx = valid_idx
    elif is_test:
        target_idx = test_idx

    gain_mean, gain_min, gain_max = calc_metrics(new_paths, teacher_paths, new_num_paths, teacher_num_paths, target_idx)

    return gain_mean, gain_min, gain_max

# Loss
for idx, method in enumerate(methods):
    if method == 'Baseline':
        continue
    InputDir = InputDirs[idx]
    loss = pd.read_csv(InputDir + '/loss.csv')
    epoch = loss['epoch'].values
    train_loss = loss['train_loss'].values
    valid_loss = loss['valid_loss'].values
    test_loss = loss['test_loss'].values

    plt.figure()
    plt.plot(epoch, train_loss, marker=".", label='train')
    plt.plot(epoch, valid_loss, marker=".", label='valid')
    plt.plot(epoch, test_loss, marker=".", label='test')
    plt.title(method)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(OutputDir + '/' + method + '_loss.pdf')

# train
result_dic = {'Method':[], 'Gain_mean':[], 'Gain_min':[], 'Gain_max':[]}
for idx, method in enumerate(methods):
    InputDir = InputDirs[idx]
    gain_mean, gain_min, gain_max = get_performance(InputDir, True, False, False)
    result_dic['Method'].append(method)
    result_dic['Gain_mean'].append(gain_mean)
    result_dic['Gain_min'].append(gain_min)
    result_dic['Gain_max'].append(gain_max)
pd.DataFrame(result_dic).to_csv(OutputDir + '/performance_train.csv')

# valid
result_dic = {'Method':[], 'Gain_mean':[], 'Gain_min':[], 'Gain_max':[]}
for idx, method in enumerate(methods):
    InputDir = InputDirs[idx]
    gain_mean, gain_min, gain_max = get_performance(InputDir, False, True, False)
    result_dic['Method'].append(method)
    result_dic['Gain_mean'].append(gain_mean)
    result_dic['Gain_min'].append(gain_min)
    result_dic['Gain_max'].append(gain_max)
pd.DataFrame(result_dic).to_csv(OutputDir + '/performance_valid.csv')

# test
result_dic = {'Method':[], 'Gain_mean':[], 'Gain_min':[], 'Gain_max':[]}
for idx, method in enumerate(methods):
    InputDir = InputDirs[idx]
    gain_mean, gain_min, gain_max = get_performance(InputDir, False, False, True)
    result_dic['Method'].append(method)
    result_dic['Gain_mean'].append(gain_mean)
    result_dic['Gain_min'].append(gain_min)
    result_dic['Gain_max'].append(gain_max)
pd.DataFrame(result_dic).to_csv(OutputDir + '/performance_test.csv')