import numpy as np
from scipy.io import mmread
import seaborn as sns
import glob
import re
import os
import sys
sns.set_style(style='whitegrid')

# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )
from setting_param import ratio_test
from setting_param import ratio_valid

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

def load_npy_data(new_paths, teacher_paths, new_num_paths, teacher_num_paths, teacher_idx_paths, node_pair_list_paths, idx):
    new = np.load(new_paths[idx])
    teacher = np.load(teacher_paths[idx])
    new_num = np.load(new_num_paths[idx])
    teacher_num = np.load(teacher_num_paths[idx])
    teacher_idx = np.load(teacher_idx_paths[idx])
    node_pair_list = np.load(node_pair_list_paths[idx])
    return new, teacher, new_num, teacher_num, teacher_idx, node_pair_list

def get_performance(InputDir, is_train, is_valid, is_test):
    new_paths, teacher_paths, new_num_paths, teacher_num_paths, teacher_idx_paths, node_pair_list_paths = data_split(InputDir)
    n_samples = len(new_paths)
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

    new_ls = []
    teacher_ls = []
    new_num_ls = []
    teacher_num_ls = []
    teacher_idx_ls = []
    node_pair_list_ls = []
    for idx in target_idx:
        new, teacher, new_num, teacher_num, teacher_idx, node_pair_list = load_npy_data(new_paths, teacher_paths, new_num_paths, teacher_num_paths, teacher_idx_paths, node_pair_list_paths, idx)
        new_ls.append(new)
        teacher_ls.append(teacher)
        new_num_ls.append(new_num)
        teacher_num_ls.append(teacher_num)
        teacher_idx_ls.append(teacher_idx)
        node_pair_list_ls.append(node_pair_list)
    return new_ls, teacher_ls, new_num_ls, teacher_num_ls, teacher_idx_ls, node_pair_list_ls

def get_predicted_new_attribute(InputDir, is_train, is_valid, is_test):
    new, teacher, new_num, teacher_num, teacher_idx, node_pair_list = get_performance(InputDir, is_train, is_valid, is_test)
    return new, teacher, new_num, teacher_num, teacher_idx, node_pair_list