import torch.nn as nn
import torch
from layers.ST_blocks import ST_blocks
from layers.GatedCNN import GatedCNN

class STGGNN(nn.Module):

    def __init__(self, opt, kernel_size=2, n_blocks=1, state_dim_bottleneck=64, annotation_dim_bottleneck=64):
        super(STGGNN, self).__init__()

        self.batchSize = opt.batchSize
        self.state_dim = opt.state_dim
        self.n_node = opt.n_node

        # ST-Block層
        self.st_blocks = ST_blocks(opt, kernel_size=kernel_size, n_blocks=n_blocks, state_dim_bottleneck=state_dim_bottleneck, annotation_dim_bottleneck=annotation_dim_bottleneck)
        opt.L = opt.L - 2 * n_blocks * (kernel_size - 1)

        # 出力層 GCNN (時間軸を単一ステップへマッピング)
        self.gcnn = GatedCNN(opt, in_channels=self.state_dim, out_channels=self.state_dim, kernel_size=opt.L)
        opt.L = 1

        # 出力層 FC (ロジスティック回帰用)
        self.out = nn.Sequential(
            nn.Linear(self.n_node, self.n_node),
            nn.Sigmoid()
        )

        opt.L = opt.init_L
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

    def forward(self, prop_state, annotation, A):
        """"
        prop_state :(batch_size, n_node, L, state_dim)
        annotation :(batch_size, n_node, L, annotation_dim)
        A          :(batch_size, 2, 3, max_nnz)
        output     :(batch_size, n_node, output_dim)
        """
        output = self.st_blocks(prop_state, annotation, A)                 # (batch_size, n_node, L-2*n_blocks(kernel_size-1), state_dim)
        if output.shape[2] > 1:
            output = self.gcnn(output)                                     # (batch_size, n_node, 1, state_dim)
        output = output.view(self.batchSize, self.n_node, self.state_dim)  # (batch_size, n_node, state_dim)
        output_batch = []
        for batch in range(self.batchSize):
            output_batch.append(self.sim_matrix(output[batch], output[batch])) # (n_node, n_node)
        output = torch.stack(output_batch, 0) # axis=0 (batch方向)にstack
        output = self.out(output)                                          # (batch_size, n_node, n_node)
        return output