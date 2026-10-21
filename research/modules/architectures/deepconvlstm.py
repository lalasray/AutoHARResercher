"""Port of the STRCWearlab Lasagne reference graph, including LSTM peepholes.

Four valid 5×1 convolutions, two peephole LSTMs, final-step classification.
Lasagne gate order (input, forget, cell, output) are retained.
This is a framework port, not the original Theano runtime/checkpoint format.
"""
import torch
from torch import nn
from .common import as_sequence


class PeepholeLSTM(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.hidden = hidden
        self.weight_ih = nn.Parameter(torch.empty(inputs, 4 * hidden))
        self.weight_hh = nn.Parameter(torch.empty(hidden, 4 * hidden))
        self.bias = nn.Parameter(torch.zeros(4 * hidden))
        self.peepholes = nn.Parameter(torch.empty(3, hidden))
        for weight in (self.weight_ih, self.weight_hh):
            for gate in weight.chunk(4, dim=1):
                nn.init.orthogonal_(gate)
        nn.init.normal_(self.peepholes, std=.1)

    def forward(self, x):
        h = x.new_zeros(x.shape[0], self.hidden)
        c = torch.zeros_like(h)
        outputs = []
        projected = x @ self.weight_ih + self.bias
        for value in projected.unbind(1):
            i, f, candidate, o = (value + h @ self.weight_hh).chunk(4, dim=-1)
            i = torch.sigmoid(i + c * self.peepholes[0])
            f = torch.sigmoid(f + c * self.peepholes[1])
            c = f * c + i * torch.tanh(candidate)
            h = torch.sigmoid(o + c * self.peepholes[2]) * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class DeepConvLSTM(nn.Module):
    def __init__(self, num_classes, params):
        super().__init__()
        self.channels = params['input_channels']
        filters, hidden = params.get('conv_filters', 64), params.get('hidden_dim', 128)
        conv = []
        incoming = 1
        for _ in range(4):
            layer = nn.Conv2d(incoming, filters, (5, 1))
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)
            conv.extend([layer, nn.ReLU()])
            incoming = filters
        self.conv = nn.Sequential(*conv)
        layers = params.get('lstm_layers', 2)
        self.lstm = nn.ModuleList([PeepholeLSTM(filters * self.channels if i == 0 else hidden, hidden)
                                  for i in range(layers)])
        self.dropout = nn.Dropout(params.get('dropout', .5))
        self.classifier = nn.Linear(hidden, num_classes)
        nn.init.orthogonal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = self.conv(as_sequence(x, self.channels).unsqueeze(1)).permute(0, 2, 1, 3).flatten(2)
        for layer in self.lstm:
            x = layer(x)
        return self.classifier(self.dropout(x[:, -1]))


def build_model(input_dim, num_classes, params):
    return DeepConvLSTM(num_classes, params)
