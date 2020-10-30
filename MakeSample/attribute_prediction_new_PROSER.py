import numpy as np
import random
import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd
import copy
import os
import sys
import networkx as nx
from scipy.io import mmread

# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )

from setting_param import MakeSample_attribute_prediction_new_PROSER_InputDir as InputDir
from setting_param import MakeSample_attribute_prediction_new_PROSER_OutputDir as OutputDir
from setting_param import Evaluation_prediction_num_of_node_new_LSTM_InputDir as predicted_num_InputDir
from setting_param import L
from setting_param import attribute_prediction_new_PROSER_threshold as threshold
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity

os.mkdir(OutputDir)
os.mkdir(OutputDir + "/input/")
os.mkdir(OutputDir + "/label/")
os.mkdir(OutputDir + "/input_num/")

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
    return  np.load(InputDir + '/node_attribute' + str(ts) + '.npy')

def TsSplit(ts, L):
    ts_train = [(ts+l) for l in range(L)]
    ts_test = ts_train[-1]+1
    ts_all = ts_train.copy()
    ts_all.extend([ts_test])
    return ts_train, ts_test, ts_all

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

for ts in range(L, EXIST_TABLE.shape[1] - L):
    ts_train, ts_test, ts_all = TsSplit(ts, L)

    # ラベルの作成
    node_attribute_new = NodeAttribute(ts_train[-1])[sorted(GetNodes(ts_train[-1], L, 'new'))]  # (921, 300)
    node_attribute_teacher = NodeAttribute(ts_test)[sorted(GetNodes(ts_test, L, 'new'))]  # (1023, 300)
    sim_matrix = cosine_similarity(node_attribute_new, node_attribute_teacher)  # (921, 1023)
    sim_matrix_min = sim_matrix.min(axis=1) > threshold
    sim_matrix_25 = np.percentile(sim_matrix, 25, axis=1) > threshold
    sim_matrix_50 = np.percentile(sim_matrix, 50, axis=1) > threshold
    sim_matrix_75 = np.percentile(sim_matrix, 75, axis=1) > threshold
    sim_matrix_max = sim_matrix.max(axis=1) > threshold
    sim_matrix_stats = np.array([sim_matrix_min, sim_matrix_25, sim_matrix_50, sim_matrix_75, sim_matrix_max]).transpose((1, 0))  # (921, 5)
    label_matrix = np.zeros((max_new_node_num, 5))
    label_matrix[:sim_matrix_stats.shape[0]] = sim_matrix_stats
    np.save(OutputDir + "/label/" + str(ts), label_matrix)

    # 入力の作成
    node_attribute_new_mean = node_attribute_new.mean(axis=0)
    node_attribute_new_mean = np.tile(node_attribute_new_mean, (node_attribute_new.shape[0], 1))
    node_attribute_new = np.append(node_attribute_new, node_attribute_new_mean, axis=1)  # (921, 600)
    input_matrix = np.zeros((max_new_node_num, node_attribute_new.shape[1]))
    input_matrix[:node_attribute_new.shape[0]] = node_attribute_new
    np.save(OutputDir + "/input/" + str(ts), input_matrix)

    # マスクの作成
    np.save(OutputDir + "/input_num/" + str(ts), node_attribute_new.shape[0])