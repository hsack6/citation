import numpy as np
import glob
import re
import os
import sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
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

class BADataset():
    def __init__(self, path, L, is_train, is_valid, is_test):
        # 入力ファイルのPATHのリストを取得
        input_new_paths = load_paths_from_dir(path + '/input/new')
        #input_lost_paths = load_paths_from_dir(path + '/input/lost')
        label_new_paths = load_paths_from_dir(path + '/label/new')
        #label_lost_paths = load_paths_from_dir(path + '/label/lost')

        # split data
        n_samples = len(label_new_paths)
        all_idx = list(range(n_samples))
        dev_idx, test_idx = dev_test_split(all_idx, n_samples, ratio_test)
        train_idx, valid_idx = dev_test_split(dev_idx, n_samples, ratio_valid)

        if is_train:
            #target_idx = train_idx
            target_idx = all_idx[-18:-4]
        elif is_valid:
            #target_idx = valid_idx
            target_idx = all_idx[-4:-2]
        elif is_test:
            #target_idx = test_idx
            target_idx = all_idx[-2:]
        else:
            target_idx = all_idx

        self.idx_list = target_idx
        self.input_new = [np.load(input_new_paths[idx]) for idx in target_idx]
        #self.input_lost = [np.load(input_lost_paths[idx]) for idx in target_idx]
        self.label_new = [np.load(label_new_paths[idx]) for idx in target_idx]
        #self.label_lost = [np.load(label_lost_paths[idx]) for idx in target_idx]
        self.L = L

    def __getitem__(self, index):
        sample_idx = self.idx_list[index] + self.L
        input_new = self.input_new[index]
        input_lost = 0
        label_new = self.label_new[index]
        label_lost = 0
        return sample_idx, input_new, input_lost, label_new, label_lost

    def __len__(self):
        return len(self.idx_list)