import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, opt):
        super(MLP, self).__init__()

        self.batchSize = opt.batchSize
        self.state_dim = opt.state_dim
        self.L = opt.L
        self.output_dim = opt.output_dim
        self.n_node = opt.n_node

        # 出力層 FC (ロジスティック回帰用)
        self.out = nn.Sequential(
            nn.Linear(self.state_dim, opt.output_dim),
            nn.Sigmoid()
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state):
        """"
        prop_state  :(batch_size, n_node, L, state_dim)
        """
        output = prop_state[:, :, -1]  # (batch_size, n_node, state_dim)
        output = self.out(output)
        return output