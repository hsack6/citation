from __future__ import print_function
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, d0):
        super(FNN, self).__init__()
        self.out = nn.Linear(d0, d0)

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        """
        x: (batchSize, n_node, d0)
        """
        x = self.out(x)  # (batchSize, n_node, d0)
        return x
