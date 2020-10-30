import numpy as np
from matplotlib import pyplot as plt
import os
import sys

# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )

from setting_param import MakeSample_prediction_num_of_node_InputDir, MakeSample_prediction_num_of_node_OutputDir
from setting_param import L
InputDir = MakeSample_prediction_num_of_node_InputDir
OutputDir = MakeSample_prediction_num_of_node_OutputDir

os.makedirs(OutputDir, exist_ok=True)
os.mkdir(OutputDir + "/input/")
os.mkdir(OutputDir + "/input/new")
os.mkdir(OutputDir + "/input/lost")
os.mkdir(OutputDir + "/label/")
os.mkdir(OutputDir + "/label/new")
os.mkdir(OutputDir + "/label/lost")

# READ EXIST_TABLE
EXIST_TABLE = np.load(InputDir + '/exist_table.npy')

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

for ts in range(L, EXIST_TABLE.shape[1]-L):
    ts_train, ts_test, ts_all = TsSplit(ts, L)
    new = []
    lost = []
    for ts_ in ts_train:
        new.append(len(GetNodes(ts_, L, 'new')))
        lost.append(len(GetNodes(ts_, L, 'lost')))
    np.save(OutputDir + "/input/new/" + str(ts), new)
    np.save(OutputDir + "/input/lost/" + str(ts), lost)
    np.save(OutputDir + "/label/new/" + str(ts), [len(GetNodes(ts_test, L, 'new'))])
    np.save(OutputDir + "/label/lost/" + str(ts), [len(GetNodes(ts_test, L, 'lost'))])

os.makedirs("statistics/num_of_nodes", exist_ok=True)

new = []
lost = []

for ts in range(L, EXIST_TABLE.shape[1]):
    new.append(len(GetNodes(ts, L, 'new')))
    lost.append(len(GetNodes(ts, L, 'lost')))

plt.figure()
plt.plot([ts for ts in range(L, EXIST_TABLE.shape[1])], new, marker=".")
plt.title('new')
plt.xlabel('time step')
plt.ylabel('# of nodes')
plt.savefig('statistics/num_of_nodes/new.pdf')

plt.figure()
plt.plot([ts for ts in range(L, EXIST_TABLE.shape[1])], lost, marker=".")
plt.title('lost')
plt.xlabel('time step')
plt.ylabel('# of nodes')
plt.savefig('statistics/num_of_nodes/lost.pdf')

print("prediction_num_of_node_new, max:" + str(max(new)) + ", min:" + str(min(new)))
print("prediction_num_of_node_lost, max:" + str(max(lost)) + ", min:" + str(min(lost)))