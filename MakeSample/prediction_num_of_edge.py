import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import networkx as nx
from scipy.io import mmwrite, mmread

# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )

from setting_param import MakeSample_prediction_num_of_edge_InputDir, MakeSample_prediction_num_of_edge_OutputDir
from setting_param import L
InputDir = MakeSample_prediction_num_of_edge_InputDir
OutputDir = MakeSample_prediction_num_of_edge_OutputDir

os.makedirs(OutputDir, exist_ok=True)
os.mkdir(OutputDir + "/input/")
os.mkdir(OutputDir + "/input/new")
os.mkdir(OutputDir + "/input/appeared")
os.mkdir(OutputDir + "/input/disappeared")
os.mkdir(OutputDir + "/label/")
os.mkdir(OutputDir + "/label/new")
os.mkdir(OutputDir + "/label/appeared")
os.mkdir(OutputDir + "/label/disappeared")

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

def Nx(ts):
    return  nx.from_numpy_matrix(mmread(InputDir + '/adjacency' + str(ts)).toarray())

def SubNxNew(ts, L):
    #return nx.Graph(Nx(ts).edges(GetNodes(ts, L, 'new')))
    matrix_new = mmread(InputDir + '/adjacency' + str(ts)) - mmread(InputDir + '/adjacency' + str(ts-1))
    return nx.from_scipy_sparse_matrix(matrix_new)

def SubNxLost(ts, L):
    return nx.Graph(Nx(ts-1).edges(GetNodes(ts, L, 'lost')))

def GetEdges(ts, L, edge_type):
    G_1 = Nx(ts)
    if edge_type == "all":
        edge_set = G_1.edges
    elif edge_type == 'stay':
        G_0 = Nx(ts - 1)
        edge_set = G_0.edges & G_1.edges
    elif edge_type == "appeared":
        G_0 = Nx(ts - 1)
        edge_set = G_1.edges - G_0.edges - SubNxNew(ts, L).edges
    elif edge_type == "disappeared":
        G_0 = Nx(ts - 1)
        edge_set = G_0.edges - G_1.edges - SubNxLost(ts, L).edges
    return edge_set

def TsSplit(ts, L):
    ts_train = [(ts+l) for l in range(L)]
    ts_test = ts_train[-1]+1
    ts_all = ts_train.copy()
    ts_all.extend([ts_test])
    return ts_train, ts_test, ts_all

for ts in range(L, EXIST_TABLE.shape[1]-L):
    ts_train, ts_test, ts_all = TsSplit(ts, L)
    new = []
    #disappeared = []
    #appeared = []
    for ts_ in ts_train:
        new.append(len(SubNxNew(ts_, L).edges))
        #disappeared.append(len(GetEdges(ts_, L, 'disappeared')))
        #appeared.append(len(GetEdges(ts_, L, 'appeared')))
    np.save(OutputDir + "/input/new/" + str(ts), new)
    #np.save(OutputDir + "/input/disappeared/" + str(ts), disappeared)
    #np.save(OutputDir + "/input/appeared/" + str(ts), appeared)
    np.save(OutputDir + "/label/new/" + str(ts), [len(SubNxNew(ts_test, L).edges)])
    #np.save(OutputDir + "/label/disappeared/" + str(ts), [len(GetEdges(ts_test, L, 'disappeared'))])
    #np.save(OutputDir + "/label/appeared/" + str(ts), [len(GetEdges(ts_test, L, 'appeared'))])

os.makedirs("statistics/num_of_edges", exist_ok=True)

new = []
#disappeared = []
#appeared = []

for ts in range(L, EXIST_TABLE.shape[1]):
    new.append(len(SubNxNew(ts, L).edges))
    #disappeared.append(len(GetEdges(ts, L, 'disappeared')))
    #appeared.append(len(GetEdges(ts, L, 'appeared')))

plt.figure()
plt.plot([ts for ts in range(L, EXIST_TABLE.shape[1])], new, marker=".")
plt.title('new')
plt.xlabel('time step')
plt.ylabel('# of edges')
plt.savefig('statistics/num_of_edges/new.pdf')

"""
plt.figure()
plt.plot([ts for ts in range(L, EXIST_TABLE.shape[1])], appeared, marker=".")
plt.title('appeared')
plt.xlabel('time step')
plt.ylabel('# of edges')
plt.savefig('statistics/num_of_edges/appeared.pdf')

plt.figure()
plt.plot([ts for ts in range(L, EXIST_TABLE.shape[1])], disappeared, marker=".")
plt.title('disappeared')
plt.xlabel('time step')
plt.ylabel('# of edges')
plt.savefig('statistics/num_of_edges/disappeared.pdf')
"""

#print("prediction_num_of_edge_appeared, max:" + str(max(appeared)) + ", min:" + str(min(appeared)))
#print("prediction_num_of_edge_disappeared, max:" + str(max(disappeared)) + ", min:" + str(min(disappeared)))
print("prediction_num_of_edge_new, max:" + str(max(new)) + ", min:" + str(min(new)))