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

from setting_param import MakeSample_attribute_prediction_new_InputDir as InputDir
from setting_param import MakeSample_attribute_prediction_new_OutputDir as OutputDir
from setting_param import Evaluation_prediction_num_of_node_new_LSTM_InputDir as predicted_num_InputDir
from setting_param import L

os.mkdir(OutputDir)
os.mkdir(OutputDir + "/new/")
os.mkdir(OutputDir + "/teacher/")
os.mkdir(OutputDir + "/new_num/")
os.mkdir(OutputDir + "/teacher_num/")
os.mkdir(OutputDir + "/teacher_idx/")

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
    return  np.load(InputDir + '/node_attribute' + str(ts) + ".npy")

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
    node_attribute = NodeAttribute(ts_train[-1])[sorted(GetNodes(ts_train[-1], L, 'new'))]
    predicted_new_node_num = int(np.load(predicted_num_InputDir + '/output/pred' + str(ts) + '.npy')[0])

    # input_matrixをnode_attributeを用いて初期化する。
    input_matrix = np.empty((0, node_attribute.shape[1]), float)
    for _ in range(predicted_new_node_num // node_attribute.shape[0]):
        input_matrix = np.append(input_matrix, node_attribute, axis=0)
    input_matrix = np.append(input_matrix, node_attribute[:predicted_new_node_num % node_attribute.shape[0]], axis=0)
    assert input_matrix.shape[0] == predicted_new_node_num, "Assignment error"
    # padding済みの行列new_matrixにinput_matrixを代入
    new_matrix = np.zeros((max_predicted_new_node_num, node_attribute.shape[1]))
    new_matrix[:predicted_new_node_num] = input_matrix

    # labelを取得
    label_matrix = NodeAttribute(ts_test)[sorted(GetNodes(ts_test, L, 'new'))]
    label_idx = sorted(GetNodes(ts_test, L, 'new'))
    # padding済みの行列teacher_matrixにlabel_matrixを代入
    teacher_matrix = np.zeros((max_new_node_num, node_attribute.shape[1]))
    teacher_matrix[:label_matrix.shape[0]] = label_matrix
    teacher_idx = np.zeros(max_new_node_num)
    teacher_idx[:len(label_idx)] = label_idx

    np.save(OutputDir + "/new/" + str(ts), new_matrix)
    np.save(OutputDir + "/new_num/" + str(ts), predicted_new_node_num)
    np.save(OutputDir + "/teacher/" + str(ts), teacher_matrix)
    np.save(OutputDir + "/teacher_num/" + str(ts), len(label_idx))
    np.save(OutputDir + "/teacher_idx/" + str(ts), teacher_idx)