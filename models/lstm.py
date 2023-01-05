from torch import nn
import torch
import torch.nn.functional as F


class PlainLSTM(nn.Module):
    def __init__(self, config, device):
        super(PlainLSTM, self).__init__()
        self.device = device
        self.config = config
        self.input_dim = 1 # size of each sequence
        self.hidden_dim = config['HIDDEN']
        self.out_dim = config['FEATURE']
        self.num_layers = config['NUM_LAYER']
        self.bidirectional = config['BIDIRECTIONAL']

        # architecture
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, 
                            num_layers=self.num_layers, batch_first=True,
                            bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), 
                            self.out_dim)

        self.use_bn = config.get("USE_BN", False)

    def init_hidden(self, x):
        h0 = torch.zeros((self.num_layers * (2 if self.bidirectional else 1), 
                         x.size(0), self.hidden_dim)).to(self.device)
        c0 = torch.zeros((self.num_layers * (2 if self.bidirectional else 1), 
                         x.size(0), self.hidden_dim)).to(self.device)
        return h0, c0

    def forward(self, x):
        # x.shape = (batch, seq, feature)
        if len(x.shape) == 2:
            x = x.unsqueeze(2)

        hidden = self.init_hidden(x)
        out, hidden = self.lstm(x, hidden)

        # last of foward (b, last, hidden)
        out_f = out[:, -1, :self.hidden_dim]
        # last of backward (b, 0, hidden:)
        out_b = out[:, 0, self.hidden_dim:]
        out = torch.cat((out_f, out_b), dim=1)
        out = self.fc(out)
        return out



class AttnLSTM(nn.Module):
    def __init__(self, config, device):
        super(AttnLSTM, self).__init__()
        self.device = device
        self.config = config
        self.input_dim = 1 # size of each sequence
        self.hidden_dim = config['HIDDEN']
        self.out_dim = config['FEATURE']
        self.num_layers = config['NUM_LAYER']
        self.bidirectional = config['BIDIRECTIONAL']

        self.dropout = nn.Dropout(p=0.5)
        # architecture
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, 
                            batch_first=True, num_layers=self.num_layers, 
                            bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), 
                            self.out_dim)

        self.w_ha = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), 
                              self.hidden_dim * (2 if self.bidirectional else 1),
                              bias=True)
        self.w_att = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1),
                               1, bias=False)

    def init_hidden(self, x):
        h0 = torch.zeros((self.num_layers * (2 if self.bidirectional else 1), 
                         x.size(0), self.hidden_dim)).to(self.device)
        c0 = torch.zeros((self.num_layers * (2 if self.bidirectional else 1), 
                         x.size(0), self.hidden_dim)).to(self.device)
        return h0, c0

    def forward(self, x):
        # x.shape = (batch, seq, feature)
        if len(x.shape) == 2:
            x = x.unsqueeze(2)

        hidden = self.init_hidden(x)
        lstm_out, hidden = self.lstm(x, hidden)        
        a_states = self.w_ha(lstm_out)
        alpha = torch.softmax(self.w_att(a_states), dim=1).view(x.size(0), 1, x.size(1))
        weighted_sum = torch.bmm(alpha, a_states)

        lat = weighted_sum.view(x.size(0), -1)
        out = self.fc(lat)
        return out
