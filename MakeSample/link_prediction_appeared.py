import numpy as np
import os
import sys
import networkx as nx
from scipy.sparse import lil_matrix
from scipy.io import mmwrite, mmread

# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )

from setting_param import MakeSample_link_prediction_appeared_InputDir as InputDir
from setting_param import MakeSample_link_prediction_appeared_OutputDir as OutputDir
from setting_param import L
from setting_param import attribute_dim

os.mkdir(OutputDir)
os.mkdir(OutputDir + "/input/")
os.mkdir(OutputDir + "/input/node_attribute/")
os.mkdir(OutputDir + "/input/adjacency")
os.mkdir(OutputDir + "/label/")
os.mkdir(OutputDir + "/mask/")

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

def Nx(ts):
    return  nx.from_numpy_matrix(mmread(InputDir + '/adjacency' + str(ts)).toarray())

def SubNxNew(ts, L):
    return nx.Graph(Nx(ts).edges(GetNodes(ts, L, 'new')))

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

def get_adjacency_matrix(ts, L, edge_type):
    G = nx.Graph(list(GetEdges(ts, L, edge_type)))
    A = np.array(nx.to_numpy_matrix(G, nodelist=[i for i in range(n_node)]))
    return A

def get_exist_matrix(ts):
    index = np.where(EXIST_TABLE[:, ts] == 1)[0]
    exist_row = np.zeros((n_node, n_node))
    exist_row[index] = 1
    exist_col = np.zeros((n_node, n_node))
    exist_col[:, index] = 1
    return exist_row * exist_col

def NodeAttribute(ts):
    return  np.load(InputDir + '/node_attribute' + str(ts) + '.npy')

def TsSplit(ts, L):
    ts_train = [(ts+l) for l in range(L)]
    ts_test = ts_train[-1]+1
    ts_all = ts_train.copy()
    ts_all.extend([ts_test])
    return ts_train, ts_test, ts_all

for ts in range(L, EXIST_TABLE.shape[1] - L):
    ts_train, ts_test, ts_all = TsSplit(ts, L)
    node_attribute = np.zeros((n_node, attribute_dim * L))
    npy_adjacency_matrix = np.zeros((n_node, n_node * L))
    for idx, ts_ in enumerate(ts_train):
        node_attribute[:, attribute_dim * idx: attribute_dim * (idx + 1)] = NodeAttribute(ts_)
        npy_adjacency_matrix[:, n_node * idx: n_node * (idx + 1)] = get_adjacency_matrix(ts_, L, 'all')

    lil_adjacency_matrix = lil_matrix(npy_adjacency_matrix)
    lil_node_attribute = lil_matrix(node_attribute)
    mmwrite(OutputDir + "/input/node_attribute/" + str(ts), lil_node_attribute)
    mmwrite(OutputDir + "/input/adjacency/" + str(ts), lil_adjacency_matrix)
    mmwrite(OutputDir + "/label/" + str(ts), lil_matrix(get_adjacency_matrix(ts_test, L, "appeared")))

    exist_matrix = get_exist_matrix(ts_train[-1])
    np.fill_diagonal(exist_matrix, 0)
    mask = exist_matrix - get_adjacency_matrix(ts_train[-1], L, 'all')
    mmwrite(OutputDir + "/mask/" + str(ts), lil_matrix(mask))
