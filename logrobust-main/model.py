import torch
import torch.nn as nn
from torch.autograd import Variable


class robustlog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys=2,device='cpu'):
        super(robustlog, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.5)
        self.num_directions = 2
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

        self.attention_size = self.hidden_size
        self.w_omega = Variable(
            torch.zeros(self.hidden_size * self.num_directions, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))
        self.sequence_length = 100

    def attention_net(self, lstm_output, device):
        output_reshape = torch.Tensor.reshape(lstm_output,
                                              [-1, self.hidden_size * self.num_directions])

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega.to(device)))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega.to(device), [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, self.sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, self.sequence_length, 1])
        state = lstm_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, features):
        inp = features
        # inp = features[2]
        self.sequence_length = inp.shape[1]
        out, _ = self.lstm(inp)
        out = self.attention_net(out, self.device)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    device = torch.device('cuda:0')
    model = robustlog(300,10,2,device=device).to(device)
    inp = torch.ones((64,20,300)).to(device)
    output = model(inp)
    print(output.shape)
