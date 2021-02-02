import numpy as np
from scipy.io import mmread
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import re
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
sns.set(style='darkgrid')
sns.set_style(style='whitegrid')

# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )
from setting_param import ratio_test
from setting_param import ratio_valid
from setting_param import all_node_num
from setting_param import MakeSample_repeat2_attribute_prediction_new_InputDir

EXIST_TABLE = np.load(MakeSample_repeat2_attribute_prediction_new_InputDir + '/exist_table.npy')

from setting_param import Evaluation_repeat2_attribute_prediction_new_utilize_new_attribute_link_Baseline_InputDir as Baseline_InputDir
from setting_param import Evaluation_repeat2_attribute_prediction_new_utilize_new_attribute_link_LSTM_InputDir as LSTM_InputDir
from setting_param import Evaluation_repeat2_attribute_prediction_new_utilize_new_attribute_link_STGGNN_InputDir as STGGNN_InputDir
from setting_param import Evaluation_repeat2_attribute_prediction_new_utilize_new_attribute_link_EGCNh_InputDir as EGCNh_InputDir
from setting_param import Evaluation_repeat2_attribute_prediction_new_utilize_new_attribute_link_STGCN_InputDir as STGCN_InputDir
from setting_param import Evaluation_repeat2_attribute_prediction_new_utilize_new_attribute_link_EGCNo_InputDir as EGCNo_InputDir
from setting_param import Evaluation_repeat2_attribute_prediction_new_utilize_new_attribute_link_GCN_InputDir as GCN_InputDir
from setting_param import Evaluation_repeat2_attribute_prediction_new_utilize_new_attribute_link_DynGEM_InputDir as DynGEM_InputDir
from setting_param import Evaluation_repeat2_attribute_prediction_new_utilize_new_attribute_link_FNN_InputDir as FNN_InputDir

from setting_param import Evaluation_repeat2_attribute_prediction_new_utilize_new_attribute_link_OutputDir as OutputDir

# InputDirs = [Baseline_InputDir, FNN_InputDir, LSTM_InputDir, GCN_InputDir, STGCN_InputDir, EGCNh_InputDir, EGCNo_InputDir, STGGNN_InputDir]
# methods = ['Baseline', 'FNN', 'LSTM', 'GCN', 'STGCN', 'EvolveGCN-H', 'EvolveGCN-O', 'TGGNN']
InputDirs = [FNN_InputDir]
methods = ['FNN']
os.makedirs(OutputDir, exist_ok=True)
os.makedirs(OutputDir + '/train', exist_ok=True)
os.makedirs(OutputDir + '/valid', exist_ok=True)
os.makedirs(OutputDir + '/test', exist_ok=True)

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

def calc_score(true_paths, pred_paths, mask_paths, target_idx):
    y_mean = []
    y_median = []
    y_min = []
    y_max = []
    for idx in target_idx:
        mask = mmread(mask_paths[idx]).toarray()[0][0]
        true = mmread(true_paths[idx]).toarray()[:mask]
        pred = mmread(pred_paths[idx]).toarray()[all_node_num : all_node_num+mask]
        cs_npy = np.diag(cosine_similarity(true, pred))
        y_mean.append(cs_npy.mean())
        y_median.append(np.median(cs_npy))
        y_min.append(cs_npy.min())
        y_max.append(cs_npy.max())
    y_mean = sum(y_mean)/len(target_idx)
    y_median = sum(y_median)/len(target_idx)
    y_min = sum(y_min)/len(target_idx)
    y_max = sum(y_max)/len(target_idx)
    return y_mean, y_median, y_min, y_max

def get_performance(InputDir, method, is_train, is_valid, is_test):
    true_paths, pred_paths, mask_paths = true_pred_mask_split(InputDir)
    n_samples = len(true_paths)
    all_idx = list(range(n_samples))
    dev_idx, test_idx = dev_test_split(all_idx, n_samples, ratio_test)
    train_idx, valid_idx = dev_test_split(dev_idx, n_samples, ratio_valid)

    if is_train:
        # target_idx = train_idx
        target_idx = all_idx[:-4]
    elif is_valid:
        # target_idx = valid_idx
        target_idx = all_idx[-4:-2]
    elif is_test:
        # target_idx = test_idx
        target_idx = all_idx[-2:]

    y_mean, y_median, y_min, y_max = calc_score(true_paths, pred_paths, mask_paths, target_idx)
    return y_mean, y_median, y_min, y_max


# Loss
for idx, method in enumerate(methods):
    if method == 'Baseline' or method == 'Random':
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

result_dic = {'Method':[], 'mean':[], 'median':[], 'min':[], 'max':[]}
for idx, method in enumerate(methods):
    InputDir = InputDirs[idx]
    y_mean, y_median, y_min, y_max = get_performance(InputDir, method, False, False, True)
    result_dic['Method'].append(method)
    result_dic['mean'].append(y_mean)
    result_dic['median'].append(y_median)
    result_dic['min'].append(y_min)
    result_dic['max'].append(y_max)
pd.DataFrame(result_dic).to_csv(OutputDir + '/performance_test.csv')