import torch.nn as nn
import torch

class LSTM(nn.Module):

    def __init__(self, opt, hidden_state):
        super(LSTM, self).__init__()

        self.batchSize = opt.batchSize
        self.state_dim = opt.state_dim
        self.L = opt.L
        self.hidden_dim = hidden_state
        self.output_dim = opt.output_dim
        self.n_node = opt.n_node

        self.lstm = nn.LSTM(input_size=self.state_dim,
                             hidden_size=self.hidden_dim,
                             batch_first=True)

        # 出力層 FC (ロジスティック回帰用)
        self.out = nn.Sequential(
            nn.Linear(self.output_dim, opt.output_dim),
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
        prop_state  :(batch_size, L, n_node, state_dim)
        input_state :(batch_size * n_node, L, state_dim)
        h_t         :(batch_size, L, hidden_dim) 最後の層の各tにおける隠れ状態
        h_n         :(num_layers * num_directions, batch_size, hidden_dim) 時系列の最後の隠れ状態
        c_n         :(num_layers * num_directions, batch_size, hidden_dim) 時系列の最後のセル状態
        num_layersはLSTMの層数、スタック数。
        num_directionsはデフォルト1、双方向で2。
        """
        input_state = prop_state.view(self.batchSize * self.n_node, self.L, self.state_dim)
        h_t, (h_n, c_n) = self.lstm(input_state)
        output = h_n[0]
        output = output.view(self.batchSize, self.n_node, self.hidden_dim)
        output_batch = []
        for batch in range(self.batchSize):
            output_batch.append(self.sim_matrix(output[batch], output[batch])) # (n_node, n_node)
        output = torch.stack(output_batch, 0) # axis=0 (batch方向)にstack
        output = self.out(output)
        output = output.view(self.batchSize, self.n_node, self.output_dim)
        return output