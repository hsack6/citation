import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, opt, hidden_state):
        super(LSTM, self).__init__()

        self.batchSize = opt.batchSize
        self.state_dim = opt.state_dim
        self.L = opt.L
        self.hidden_dim = hidden_state

        self.lstm = nn.LSTM(input_size=self.state_dim,
                             hidden_size=self.hidden_dim,
                             batch_first=True)
        self.out = nn.Linear(self.hidden_dim, opt.output_dim)

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state):
        """"
        prop_state  :(batch_size, L)
        input_state :(batch_size, L, state_dim)
        h_t         :(batch_size, L, hidden_dim) 最後の層の各tにおける隠れ状態
        h_n         :(num_layers * num_directions, batch_size, hidden_dim) 時系列の最後の隠れ状態
        c_n         :(num_layers * num_directions, batch_size, hidden_dim) 時系列の最後のセル状態
        num_layersはLSTMの層数、スタック数。
        num_directionsはデフォルト1、双方向で2。
        """
        input_state = prop_state.view(self.batchSize, self.L, self.state_dim)
        h_t, (h_n, c_n) = self.lstm(input_state)
        h_n = h_n[0]
        output = self.out(h_n)
        return output