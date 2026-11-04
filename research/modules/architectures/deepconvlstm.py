"""DeepConvLSTM adaptation using a fused PyTorch LSTM (cuDNN on CUDA).

Four valid 5x1 convolutions and final-step classification are retained.
Standard LSTM gates replace the reference peephole recurrence.
"""
from torch import nn
from .common import as_sequence


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
        self.lstm = nn.LSTM(filters * self.channels, hidden, num_layers=layers,
                            batch_first=True)
        for name, parameter in self.lstm.named_parameters():
            if name.startswith('weight_'):
                for gate in parameter.chunk(4, dim=0):
                    nn.init.orthogonal_(gate)
            else:
                nn.init.zeros_(parameter)
        self.dropout = nn.Dropout(params.get('dropout', .5))
        self.classifier = nn.Linear(hidden, num_classes)
        nn.init.orthogonal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = self.conv(as_sequence(x, self.channels).unsqueeze(1)).permute(0, 2, 1, 3).flatten(2)
        x, _ = self.lstm(x.contiguous())
        return self.classifier(self.dropout(x[:, -1]))


def build_model(input_dim, num_classes, params):
    return DeepConvLSTM(num_classes, params)
