import numpy as np
from scipy.io import mmread
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import re
import os
import sys
import random
from sklearn.metrics import roc_curve, roc_auc_score
sns.set(style='darkgrid')
sns.set_style(style='whitegrid')

# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )
from setting_param import ratio_test
from setting_param import ratio_valid
from setting_param import MakeSample_link_prediction_new_InputDir

EXIST_TABLE = np.load(MakeSample_link_prediction_new_InputDir + '/exist_table.npy')

from setting_param import Evaluation_link_prediction_new_COSSIMMLP_Baseline_mix_InputDir as COSSIMMLP_Baseline_mix_InputDir
from setting_param import Evaluation_link_prediction_new_COSSIMMLP_Baseline_learning_InputDir as COSSIMMLP_Baseline_learning_InputDir
from setting_param import Evaluation_link_prediction_new_COSSIMMLP_Baseline_inference_InputDir as COSSIMMLP_Baseline_inference_InputDir

from setting_param import Evaluation_link_prediction_new_COSSIMMLP_DeepMatchMax_mix_InputDir as COSSIMMLP_DeepMatchMax_mix_InputDir
from setting_param import Evaluation_link_prediction_new_COSSIMMLP_DeepMatchMax_learning_InputDir as COSSIMMLP_DeepMatchMax_learning_InputDir
from setting_param import Evaluation_link_prediction_new_COSSIMMLP_DeepMatchMax_inference_InputDir as COSSIMMLP_DeepMatchMax_inference_InputDir

from setting_param import Evaluation_link_prediction_new_COSSIMMLP_FNN_mix_InputDir as COSSIMMLP_FNN_mix_InputDir
from setting_param import Evaluation_link_prediction_new_COSSIMMLP_FNN_learning_InputDir as COSSIMMLP_FNN_learning_InputDir
from setting_param import Evaluation_link_prediction_new_COSSIMMLP_FNN_inference_InputDir as COSSIMMLP_FNN_inference_InputDir


from setting_param import Evaluation_link_prediction_new_OutputDir as OutputDir

InputDirs = [COSSIMMLP_Baseline_learning_InputDir, COSSIMMLP_Baseline_inference_InputDir, COSSIMMLP_Baseline_mix_InputDir, COSSIMMLP_DeepMatchMax_learning_InputDir, COSSIMMLP_DeepMatchMax_inference_InputDir, COSSIMMLP_DeepMatchMax_mix_InputDir, COSSIMMLP_FNN_learning_InputDir, COSSIMMLP_FNN_inference_InputDir, COSSIMMLP_FNN_mix_InputDir]
methods = ['COSSIMMLP_Baseline_learning', 'COSSIMMLP_Baseline_inference', 'COSSIMMLP_Baseline_mix', 'COSSIMMLP_DeepMatchMax_learning', 'COSSIMMLP_DeepMatchMax_inference', 'COSSIMMLP_DeepMatchMax_mix', 'COSSIMMLP_FNN_learning', 'COSSIMMLP_FNN_inference', 'COSSIMMLP_FNN_mix']
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

def _random_subset(seq,m):
    targets=set()
    while len(targets)<m:
        x=random.choice(seq)
        targets.add(x)
    return targets

def balancer(target, mask):
    target = target[0 < mask]
    n_positive = int(target.sum())
    n_negative = int(len(target) - n_positive)
    if n_positive <= n_negative:
        sample_idx_list = np.where(target==1)[0].tolist()
        negative_idx = np.where(target==0)[0]
        sample_idx_list.extend(list(_random_subset(negative_idx, n_positive)))
    else:
        sample_idx_list = np.where(target==0)[0].tolist()
        positive_idx = np.where(target==1)[0]
        sample_idx_list.extend(list(_random_subset(positive_idx, n_negative)))
    return sample_idx_list

def calc_roc(true_paths, pred_paths, mask_paths, target_idx):
    y_true = []
    y_pred = []
    for idx in target_idx:
        true = mmread(true_paths[idx]).toarray()
        pred = mmread(pred_paths[idx]).toarray()
        mask = mmread(mask_paths[idx]).toarray()
        sample_idx_list = balancer(true, mask)
        # posとnegでバランスを取る場合
        # y_true.append(true[0 < mask][sample_idx_list].tolist())
        # y_pred.append(pred[0 < mask][sample_idx_list].tolist())
        # posとnegでバランスを取らない場合
        y_true.append(true[0 < mask].tolist())
        y_pred.append(pred[0 < mask].tolist())
    y_true = sum(y_true, [])
    y_pred = sum(y_pred, [])
    if sum(y_true) == 0:
        print("positiveラベルが一つもないのでaucを計算できない")
    return roc_curve(y_true, y_pred), roc_auc_score(y_true, y_pred)

def get_performance(InputDir, method, is_train, is_valid, is_test):
    true_paths, pred_paths, mask_paths = true_pred_mask_split(InputDir)
    n_samples = len(true_paths)
    all_idx = list(range(n_samples))
    dev_idx, test_idx = dev_test_split(all_idx, n_samples, ratio_test)
    train_idx, valid_idx = dev_test_split(dev_idx, n_samples, ratio_valid)

    if is_train:
        target_idx = train_idx
    elif is_valid:
        target_idx = valid_idx
    elif is_test:
        target_idx = test_idx

    (fpr, tpr, thresholds), auc = calc_roc(true_paths, pred_paths, mask_paths, target_idx)
    return fpr, tpr, thresholds, auc

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

for idx, method in enumerate(methods):
    InputDir = InputDirs[idx]
    fpr, tpr, thresholds, auc = get_performance(InputDir, method, True, False, False)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.title('auc = ' + str(auc))
    plt.savefig(OutputDir + '/train/roc_curve_' + method + '.pdf')

for idx, method in enumerate(methods):
    InputDir = InputDirs[idx]
    fpr, tpr, thresholds, auc = get_performance(InputDir, method, False, True, False)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.title('auc = ' + str(auc))
    plt.savefig(OutputDir + '/valid/roc_curve_' + method + '.pdf')

for idx, method in enumerate(methods):
    InputDir = InputDirs[idx]
    fpr, tpr, thresholds, auc = get_performance(InputDir, method, False, False, True)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.title('auc = ' + str(auc))
    plt.savefig(OutputDir + '/test/roc_curve_' + method + '.pdf')