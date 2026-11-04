import math

import torch
from torch import nn
from torch.nn import functional as F


class Context(nn.Module):
    def __init__(self, channels, hidden=100, depth=4):
        super().__init__()
        self.project = nn.Linear(channels, hidden)
        self.token = nn.Parameter(torch.randn(1, 1, hidden))
        layer = nn.TransformerEncoderLayer(hidden, 4, 64, .1, activation='relu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)

    def forward(self, x):
        x = self.project(x)
        return self.encoder(torch.cat([self.token.expand(len(x), -1, -1), x], 1))[:, 0]


class CATCC(nn.Module):
    def __init__(self, channels, classes, length, options):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 32, 8, padding=4, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2, padding=1), nn.Dropout(.35),
            nn.Conv1d(32, 64, 8, padding=4, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, padding=1),
            nn.Conv1d(64, 128, 8, padding=4, bias=False), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2, padding=1))
        self.encoder.eval()
        with torch.no_grad():
            shape = self.encoder(torch.zeros(2, channels, length)).shape
        self.head = nn.Linear(shape[1] * shape[2], classes)
        self.horizon = options['temporal_steps']
        if self.horizon >= shape[2]:
            raise ValueError('CA-TCC window must produce more feature steps than temporal_steps')
        self.context = Context(128)
        self.predictors = nn.ModuleList([nn.Linear(100, 128) for _ in range(self.horizon)])
        self.projector = nn.Sequential(nn.Linear(100, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 32))

    def features(self, x):
        return self.encoder(x.transpose(1, 2))

    def forward(self, x):
        return self.head(self.features(x).flatten(1))

    def temporal(self, a, b):
        a, b = F.normalize(a, dim=1).transpose(1, 2), F.normalize(b, dim=1).transpose(1, 2)
        t = int(torch.randint(a.shape[1] - self.horizon, (), device=a.device))
        context = self.context(a[:, :t+1])
        losses = []
        target = torch.arange(len(a), device=a.device)
        for k, predictor in enumerate(self.predictors):

            scores = b[:, t+k+1] @ predictor(context).T
            losses.append(F.cross_entropy(scores, target))
        return torch.stack(losses).mean(), self.projector(context)


class SDMix(nn.Module):
    def __init__(self, channels, classes, length, options):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 16, 9), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 9), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.encoder.eval()
        with torch.no_grad():
            size = self.encoder(torch.zeros(2, channels, length)).numel() // 2
        self.bottleneck = nn.Sequential(nn.Linear(size, options['bottleneck']), nn.BatchNorm1d(options['bottleneck']))
        self.head = nn.Linear(options['bottleneck'], classes)

    def features(self, x):
        return self.bottleneck(self.encoder(x.transpose(1, 2)).flatten(1))

    def forward(self, x):
        return self.head(self.features(x))


class CrossHAR(nn.Module):
    def __init__(self, channels, classes, length, options):
        super().__init__()
        self.project = nn.Linear(channels, 72)
        self.position = nn.Parameter(torch.randn(1, length, 72) * .02)
        self.input_norm = nn.LayerNorm(72, eps=1e-12)
        layer = nn.TransformerEncoderLayer(72, 4, 144, 0., activation='gelu', batch_first=True,
                                           layer_norm_eps=1e-12)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.decoder = nn.Sequential(nn.Linear(72,72), nn.GELU(), nn.LayerNorm(72,eps=1e-12), nn.Linear(72,channels))
        self.context = Context(72, depth=1)
        self.projector = nn.Sequential(nn.Linear(100,36), nn.BatchNorm1d(36), nn.ReLU(), nn.Linear(36,18))
        self.class_project = nn.Linear(72, 100)
        layer = nn.TransformerEncoderLayer(100,4,2048,.1,batch_first=True)
        self.classifier = nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.head = nn.Linear(100, classes)
        t = torch.arange(length)[:,None]
        scale = torch.exp(torch.arange(0,100,2) * (-math.log(10000.)/100))
        position = torch.zeros(length,100)
        position[:,0::2], position[:,1::2] = torch.sin(t*scale), torch.cos(t*scale)
        self.register_buffer('class_position', position[None])

    @staticmethod
    def normalize(x):
        return (x-x.mean(1,keepdim=True)) / (x.var(1,keepdim=True,unbiased=False)+1e-5).sqrt()

    def features(self, x):
        return self.encoder(self.input_norm(self.project(x)+self.position[:, :x.shape[1]]))

    def forward(self, x):
        encoded = self.features(self.normalize(x))
        hidden = self.class_project(encoded)+self.class_position[:, :x.shape[1]]
        return self.head(self.classifier(hidden).mean(1))


def build(method, input_dim, classes, params):
    from .settings import settings
    options = settings(method, params.get('custom_params', {}))
    channels = params['input_channels']
    return {'contratsc': CATCC, 'sdmix': SDMix, 'crosshar': CrossHAR}[method](
        channels, classes, input_dim // channels, options)
