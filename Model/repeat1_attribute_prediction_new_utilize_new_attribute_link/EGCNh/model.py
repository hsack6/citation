import egcn_utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import os
import sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
from setting_param import L
from setting_param import all_node_num_expanded as all_node_num

def get_A_in_per_batch(A, batch):
    A_in = A[batch][0]  # (3, max_nnz)
    nnz = int(A_in[0].sum().item())
    A_in = A_in[:, :nnz]  # (3, nnz)
    return A_in

def get_adj_per_t(A_in, t):
    col = A_in[2]
    idx = (t * all_node_num) <= col
    idx = idx * (col < ((t + 1) * all_node_num))
    A_t = A_in[:, idx]  # (3, nnz_per_t)
    A_t[2] = A_t[2] % all_node_num  # adjacency matrix per t
    return A_t

def get_cur_adj(A_t):
    cur_adj = {}
    cur_adj['vals'] = A_t[0]  # (nnz_per_t, )
    cur_adj['idx'] = A_t[1:].t().long()  # (nnz_per_t, 2)
    return cur_adj

def get_node_mask(cur_adj, num_nodes):
    mask = torch.zeros(num_nodes) - float("Inf")
    non_zero = cur_adj['idx'].unique()
    mask[non_zero] = 0
    return mask

def make_sparse_eye(size):
    eye_idx = torch.arange(size)
    eye_idx = torch.stack([eye_idx,eye_idx],dim=1).t()
    vals = torch.ones(size)
    eye = torch.sparse.FloatTensor(eye_idx,vals,torch.Size([size,size]))
    return eye

def normalize_adj(adj, num_nodes):
    '''
    takes an adj matrix as a dict with idx and vals and normalize it by:
        - adding an identity matrix,
        - computing the degree vector
        - multiplying each element of the adj matrix (aij) by (di*dj)^-1/2
    論文中のAから¥hat{A}への変換操作(ノード次数が高いもの同士の繋がりは価値(vals)を下げるようにする)
    単位行列足した後に、各行の合計値が1となるような正規化
    '''
    idx = adj['idx']
    vals = adj['vals']
    sp_tensor = torch.sparse.FloatTensor(idx.t(), vals.type(torch.float), torch.Size([num_nodes, num_nodes]))
    sparse_eye = make_sparse_eye(num_nodes)
    sp_tensor = sparse_eye + sp_tensor
    idx = sp_tensor._indices()
    vals = sp_tensor._values()
    degree = torch.sparse.sum(sp_tensor, dim=1).to_dense()
    di = degree[idx[0]]
    dj = degree[idx[1]]
    vals = vals * ((di * dj) ** -0.5)
    return {'idx': idx.t(), 'vals': vals}

def make_sparse_tensor(adj,tensor_type,torch_size):
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size*2)

    if tensor_type == 'float':
        test = torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
        return torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
    elif tensor_type == 'long':
        return torch.sparse.LongTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.long),
                                      tensor_size)
    else:
        raise NotImplementedError('only make floats or long sparse tensors')

def sparse_prepare_tensor(tensor,torch_size):
    tensor = make_sparse_tensor(tensor,
                                tensor_type = 'float',
                                torch_size = torch_size)
    return tensor

class EGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False):
        super().__init__()
        GRCU_args = u.Namespace({})

        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()

        # 出力層 FC (ロジスティック回帰用)
        self.FC = nn.Sequential(
            nn.Linear(args.layer_2_feats, args.output_dim)
        )

        for i in range(1, len(feats)):
            GRCU_args = u.Namespace({'in_feats': feats[i - 1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            grcu_i = GRCU(GRCU_args)
            # print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self, prop_state, annotation, A):
        """"
        prop_state :(batch_size, n_node, L, state_dim)
        annotation :(batch_size, n_node, L, annotation_dim)
        A          :(batch_size, 2, 3, max_nnz)
        output     :(batch_size, n_node, output_dim)
        """

        batch_list = []
        for batch in range(A.shape[0]):
            A_in = get_A_in_per_batch(A, batch)
            A_list = []
            Nodes_list = []
            nodes_mask_list = []
            for t in range(L):
                A_t = get_adj_per_t(A_in, t)
                cur_adj = get_cur_adj(A_t)
                node_mask = get_node_mask(cur_adj, all_node_num).unsqueeze(-1)
                node_feats = prop_state[batch, :, t,]
                cur_adj = normalize_adj(adj=cur_adj, num_nodes=all_node_num)
                adj = sparse_prepare_tensor(cur_adj, torch_size=[all_node_num])
                A_list.append(adj)
                Nodes_list.append(node_feats.float())
                nodes_mask_list.append(node_mask)

            node_feats = Nodes_list[-1]
            for unit in self.GRCU_layers:
                Nodes_list = unit(A_list, Nodes_list, nodes_mask_list)

            out = Nodes_list[-1]
            if self.skipfeats:
                out = torch.cat((out, node_feats), dim=1)  # use node_feats.to_dense() if 2hot encoded input

            out = self.FC(out)
            batch_list.append(out)

        out = torch.stack(batch_list, 0)
        return out


class GRCU(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats, self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, A_list, node_embs_list, mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            # first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights, node_embs, mask_list[t])
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq


class mat_GRU_cell(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                  args.cols,
                                  torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())

        self.choose_topk = TopK(feats=args.rows,
                                k=args.cols)

    def forward(self, prev_Q, prev_Z, mask):
        z_topk = self.choose_topk(prev_Z, mask)

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_GRU_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows, cols))

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out


class TopK(torch.nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        # Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs, mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
                isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()