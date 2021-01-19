import torch.nn as nn
import torch

import sys
import os
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
from setting_param import all_node_num
from setting_param import n_expanded


class COSSIMMLP(nn.Module):

    def __init__(self, opt):
        super(COSSIMMLP, self).__init__()

        self.batchSize = opt.batchSize
        self.state_dim = opt.state_dim

        self.out = nn.Sequential(
            nn.Sigmoid()
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        #a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None] # 行列で入力される場合
        a_n, b_n = a.norm(dim=0), b.norm(dim=0)  # 一行ずつ入力される場合
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        #sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1)) # いらないものまで計算に含まれるのでダメ
        #sim_mt = (a_norm * b_norm).sum(1) # 一度に全部計算するとメモリエラーになるのでダメ
        """
        # 入力が行列の場合
        sim_mt = torch.zeros(a_norm.shape[0], dtype=torch.double)
        for i in range(a_norm.shape[0]):
            sim_mt[i] = (a_norm[i] * b_norm[i]).sum()
        """
        sim_mt = (a_norm * b_norm).sum() # 一行ずつ入力される場合
        return sim_mt

    def forward(self, prop_state, indic, mask):
        """
        prop_state :(batch_size, n_node, state_dim)
        indic      :(batch_size, n_expanded * (all_node_num + n_expanded), 2)
        mask       :(batch_size, nnz (mask for undirected graph), 2)
        """
        output = []
        for batch in range(self.batchSize):
            pred_i = [[], []]
            pred_v = []
            for i in range(mask[batch].shape[0]):
                sim = self.sim_matrix(prop_state[batch][mask[batch][i, 0]], prop_state[batch][mask[batch][i, 1]])
                pred_i[0].append(mask[batch][i, 0].item())
                pred_i[1].append(mask[batch][i, 1].item())
                pred_v.append(self.out(sim))
                pred_i[1].append(mask[batch][i, 0].item())
                pred_i[0].append(mask[batch][i, 1].item())
                pred_v.append(self.out(sim))
            pred = torch.sparse.FloatTensor(torch.LongTensor(pred_i), torch.FloatTensor(pred_v), torch.Size([all_node_num+n_expanded, all_node_num+n_expanded]))
            output.append(pred)
            """
            既知ノードの関係ない座標を全て含んでおり無駄な計算が多く効率悪い
            output.append(self.sim_matrix(prop_state[batch], prop_state[batch])) # (n_node, n_node)
            """

            """
            indic全てを計算するのは無駄が多い
            pred_ = self.sim_matrix(prop_state[batch][indic[batch][:, 0]], prop_state[batch][indic[batch][:, 1]]) # (n_expanded * (all_node_num + n_expanded))
            pred = torch.zeros((all_node_num + n_expanded, all_node_num + n_expanded), dtype=torch.double)
            pred[all_node_num:] = pred_.reshape(n_expanded, all_node_num + n_expanded)
            output.append(pred) # (n_node, n_node)
            """

            """
            密行列だとメモリにのりきらない
            pred = torch.zeros((all_node_num + n_expanded, all_node_num + n_expanded), dtype=torch.double)
            pred[mask[batch][:, 0], mask[batch][:, 1]] = self.sim_matrix(prop_state[batch][mask[batch][:, 0]], prop_state[batch][mask[batch][:, 1]])
            pred_between_new_nodes = pred[all_node_num:, all_node_num:]
            pred = pred + pred.T
            pred[all_node_num:, all_node_num:] = pred_between_new_nodes
            output.append(pred) # (n_node, n_node)
            """
        """
        output = torch.stack(output, 0) # axis=0 (batch方向)にstack
        for batch in range(self.batchSize):
            for node in range(all_node_num + n_expanded):
                output[batch][node] = self.out(output[batch][node]) # (batch_size, n_node, n_node)
        """
        return output