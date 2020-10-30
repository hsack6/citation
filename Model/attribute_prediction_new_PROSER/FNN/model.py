from __future__ import print_function
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, opt):
        super(FNN, self).__init__()
        self.state_dim = opt.state_dim
        self.mlp0 = nn.Conv1d(opt.state_dim, opt.output_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        """
        x: (batchSize, n_node, state_dim)
        """
        x = x.transpose(2, 1).contiguous()
        x = self.sigmoid(self.mlp0(x))
        x = x.transpose(2, 1).contiguous()
        return x
