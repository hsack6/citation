import numpy as np
import os
import sys

# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )

from setting_param import MakeSample_link_prediction_new_InputDir as InputDir
from setting_param import Evaluation_prediction_num_of_node_new_LSTM_InputDir as predicted_num_InputDir

from setting_param import Model_attribute_prediction_new_Baseline_OutputDir as Baseline_Out_InputDir
from setting_param import Model_attribute_prediction_new_DeepMatchMax_OutputDir as DeepMatchMax_Out_InputDir
from setting_param import Model_attribute_prediction_new_FNN_OutputDir as FNN_Out_InputDir

from setting_param import MakeSample_link_prediction_new_Baseline_OutputDir as Baseline_OutputDir
from setting_param import MakeSample_link_prediction_new_DeepMatchMax_OutputDir as DeepMatchMax_OutputDir
from setting_param import MakeSample_link_prediction_new_FNN_OutputDir as FNN_OutputDir

Method_list = ["Baseline", "DeepMatchMax", "FNN"]
Out_InputDir_list = [Baseline_Out_InputDir, DeepMatchMax_Out_InputDir, FNN_Out_InputDir]
OutputDir_list = [Baseline_OutputDir, DeepMatchMax_OutputDir, FNN_OutputDir]

from setting_param import L

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

n_expanded = max([max_predicted_new_node_num, max_new_node_num])
print("n_expande: ", n_expanded)