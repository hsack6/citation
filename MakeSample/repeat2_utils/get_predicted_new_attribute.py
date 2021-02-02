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
from setting_param import all_node_num

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

def load_npy_data(true_paths, pred_paths, mask_paths, idx):
    mask = mmread(mask_paths[idx]).toarray()[0][0]
    true = mmread(true_paths[idx]).toarray()[:mask]
    pred = mmread(pred_paths[idx]).toarray()[all_node_num: all_node_num + mask]
    return true, pred, mask

def get_performance(InputDir, is_train, is_valid, is_test):
    true_paths, pred_paths, mask_paths = data_split(InputDir)
    n_samples = len(pred_paths)
    all_idx = list(range(n_samples))
    # dev_idx, test_idx = dev_test_split(all_idx, len(all_idx), ratio_test)
    # train_idx, valid_idx = train_valid_split(dev_idx, len(all_idx), ratio_valid)
    train_idx = all_idx[:-4]
    valid_idx = all_idx[-4:-2]
    test_idx = all_idx[-2:]

    if is_train:
        target_idx = train_idx
    elif is_valid:
        target_idx = valid_idx
    elif is_test:
        target_idx = test_idx

    true_ls = []
    pred_ls = []
    mask_ls = []
    for idx in target_idx:
        true, pred, mask = load_npy_data(true_paths, pred_paths, mask_paths, idx)
        true_ls.append(true)
        pred_ls.append(pred)
        mask_ls.append(mask)
    return true_ls, pred_ls, mask_ls

def get_predicted_new_attribute(InputDir, is_train, is_valid, is_test):
    _, new, _ = get_performance(InputDir, is_train, is_valid, is_test)
    return new