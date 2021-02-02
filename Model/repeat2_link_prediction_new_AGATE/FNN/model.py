import torch.nn as nn
import torch

class FNN(nn.Module):

    def __init__(self, opt):
        super(FNN, self).__init__()

        self.batchSize = opt.batchSize
        self.state_dim = opt.state_dim
        self.n_node = opt.n_node

        self.mlp = nn.Linear(self.state_dim, self.state_dim)

        self.out = nn.Sequential(
            nn.Linear(self.n_node, self.n_node),
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
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def forward(self, prop_state):
        """"
        prop_state :(batch_size, n_node, state_dim)
        """
        output = prop_state
        output = self.mlp(output)
        output_batch = []
        for batch in range(self.batchSize):
            output_batch.append(self.sim_matrix(output[batch], output[batch])) # (n_node, n_node)
        output = torch.stack(output_batch, 0) # axis=0 (batch方向)にstack
        output = self.out(output) # (batch_size, n_node, n_node)
        return output