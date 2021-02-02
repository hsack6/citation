import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):

    def __init__(self, opt):
        super(GCN, self).__init__()

        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.n_edge_types = opt.n_edge_types
        self.n_node = opt.n_node
        self.n_steps = opt.n_steps
        self.L = opt.L
        self.batchSize = opt.batchSize

        self.gcn_weight = nn.Linear(self.state_dim, self.state_dim)

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def matmul_A_H(self, A, prop_state):
        """"
        prop_state  :(batch_size, n_node, L, state_dim)
        A           :(batch_size, n_node, n_node) *sparse
        """
        AH_batch = []
        for batch in range(A.shape[0]):
            AH_t = []
            for t in range(self.L):
                AH_t.append(torch.sparse.mm(A[batch], prop_state[batch, :, t].float()))
            AH_batch.append(torch.stack(AH_t, 1))
        AH = torch.stack(AH_batch, 0)
        return AH

    def forward(self, prop_state, A):
        """"
        prop_state  :(batch_size, n_node, L, state_dim)
        A           :(batch_size, n_node, n_node) *sparse
        AH          :(batch_size, n_node, L, state_dim)
        output_data :(batch_size, n_node, L, state_dim)
        """
        AH = self.matmul_A_H(A, prop_state)
        AHW = self.gcn_weight(AH.double())
        output_data = F.relu(AHW)
        return output_data