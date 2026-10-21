"""PyTorch ports of author Keras operations (channels-last interfaces).

Preserve Keras population-variance batch normalization and its moving-average
convention, per-head attention dimensions, and the released liteFormer reshape.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class BatchNorm(nn.Module):
    def __init__(self, width, eps=.001, momentum=.99):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.register_buffer('running_mean', torch.zeros(width))
        self.register_buffer('running_var', torch.ones(width))
        self.eps, self.momentum = eps, momentum

    def forward(self, x):
        axes = tuple(range(x.ndim - 1))
        if self.training:
            mean = x.mean(axes)
            var = x.var(axes, unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(mean, 1-self.momentum)
                self.running_var.lerp_(var, 1-self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        return (x-mean) * torch.rsqrt(var+self.eps) * self.weight + self.bias


class Conv(nn.Module):
    def __init__(self, inc, out, kernel=3, stride=1, same=True, bias=True, groups=1, swish=False):
        super().__init__()
        self.op = nn.Conv1d(inc, out, kernel, stride=stride, groups=groups, bias=bias)
        nn.init.xavier_uniform_(self.op.weight)
        if bias:
            nn.init.zeros_(self.op.bias)
        self.same, self.swish = same, swish

    def forward(self, x):
        x = x.transpose(1, 2)
        k, s = self.op.kernel_size[0], self.op.stride[0]
        if self.same:
            pad = max(0, (math.ceil(x.shape[-1]/s)-1)*s+k-x.shape[-1])
            x = F.pad(x, (pad//2, pad-pad//2))
        x = self.op(x).transpose(1, 2)
        return F.silu(x) if self.swish else x


def linear(inc, out):
    layer = nn.Linear(inc, out)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or not self.probability:
            return x
        keep = 1-self.probability
        mask = (torch.rand((x.shape[0],)+(1,)*(x.ndim-1), device=x.device) < keep).to(x.dtype)
        return x * mask / keep


class Attention(nn.Module):
    def __init__(self, width, heads, keydim, dropout):
        super().__init__()
        self.heads, self.keydim = heads, keydim
        self.q = linear(width, heads*keydim)
        self.k = linear(width, heads*keydim)
        self.v = linear(width, heads*keydim)
        self.out = linear(heads*keydim, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b,t,_ = x.shape
        q,k,v = [layer(x).reshape(b,t,self.heads,self.keydim).transpose(1,2)
                 for layer in (self.q,self.k,self.v)]
        weights = self.dropout((q @ k.transpose(-1,-2) / math.sqrt(self.keydim)).softmax(-1))
        return self.out((weights @ v).transpose(1,2).reshape(b,t,-1))


class LiteFormer(nn.Module):
    def __init__(self, width, heads, kernel, drop):
        super().__init__()
        self.width, self.heads, self.kernel = width, heads, kernel
        self.weight = nn.Parameter(torch.empty(heads, 1, kernel))
        nn.init.uniform_(self.weight, -math.sqrt(3/kernel), math.sqrt(3/kernel))
        self.drop = DropPath(drop)

    def forward(self, x):
        b,t,_ = x.shape
        # Upstream assigns softmax to the variables on every training call.
        if self.training:
            with torch.no_grad():
                self.weight.copy_(self.weight.softmax(-1))
        z = x.reshape(-1,t,self.heads).transpose(1,2)
        z = F.conv1d(F.pad(z, ((self.kernel-1)//2,self.kernel//2)),
                     self.weight, groups=self.heads)
        # Upstream stacks heads at axis 2, then applies DropPath to this
        # expanded batch, before folding it back into original B,T,C.
        return self.drop(z.transpose(1,2).unsqueeze(-1)).reshape(b,t,self.width)


class Block(nn.Module):
    def __init__(self, width, dropout=.3, drop=0., kernel=None, heads=2, lite_heads=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.sensor = kernel is not None
        if self.sensor:
            q = width//4
            self.acc = Attention(q, heads, q, dropout)
            self.gyro = Attention(q, heads, q, dropout)
            self.lite = LiteFormer(width//2, lite_heads, kernel, drop)
        else:
            self.attention = Attention(width, heads, width, dropout)
        self.fc1, self.fc2 = linear(width,width*2), linear(width*2,width)
        self.dropout = nn.Dropout(dropout)
        self.path = DropPath(drop)

    def forward(self, x):
        z = self.norm1(x)
        if self.sensor:
            q = x.shape[-1]//4
            z = torch.cat((self.path(self.acc(z[:,:,:q])), self.lite(z[:,:,q:3*q]),
                           self.path(self.gyro(z[:,:,3*q:]))), dim=-1)
        else:
            z = self.attention(z)
        x = x+z
        return x+self.path(self.fc2(self.dropout(F.silu(self.fc1(self.norm2(x))))))


class HART(nn.Module):
    def __init__(self, steps, classes, width=192, dropout=.3):
        super().__init__()
        if width % 8 or steps < 16:
            raise ValueError('HART requires width divisible by 8 and at least 16 samples')
        self.acc, self.gyro = Conv(3,width//2,16,16,False), Conv(3,width//2,16,16,False)
        self.position = nn.Parameter(torch.empty(steps//16,width))
        nn.init.uniform_(self.position,-.05,.05)
        self.blocks = nn.ModuleList([Block(width,dropout,dropout*i/5,k,3,4)
                                     for i,k in enumerate([3,7,15,31,31,31])])
        self.norm = nn.LayerNorm(width,eps=1e-6)
        self.head, self.output = linear(width,1024), linear(1024,classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = torch.cat((self.acc(x[:,:,:3]),self.gyro(x[:,:,3:])),dim=-1)+self.position
        for block in self.blocks:
            x=block(x)
        return self.output(self.dropout(F.silu(self.head(self.norm(x).mean(1)))))


class Inverted(nn.Module):
    def __init__(self, inc, expanded, out, stride=1):
        super().__init__()
        self.expand=Conv(inc,expanded,1,bias=False)
        self.bn1=BatchNorm(expanded)
        self.depth=Conv(expanded,expanded,3,stride,same=stride==1,bias=False,groups=expanded)
        self.bn2=BatchNorm(expanded)
        self.project=Conv(expanded,out,1,bias=False)
        self.bn3=BatchNorm(out)
        self.stride, self.residual = stride, stride==1 and inc==out

    def forward(self,x):
        z=F.silu(self.bn1(self.expand(x)))
        if self.stride==2:
            z=F.pad(z,(0,0,1,1))
        z=F.silu(self.bn2(self.depth(z)))
        z=self.bn3(self.project(z))
        return x+z if self.residual else z


class SensorFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.local=nn.ModuleList([nn.Sequential(Conv(32,48,swish=True),Conv(48,48,1,swish=True)) for _ in range(2)])
        self.blocks=nn.Sequential(Block(96,.3,0,4),Block(96,.3,.3,4))
        self.fold=nn.ModuleList([Conv(48,32,1,swish=True) for _ in range(2)])
        self.fuse=nn.ModuleList([Conv(64,48,swish=True) for _ in range(2)])

    def forward(self,a,g):
        z=self.blocks(torch.cat([m(x) for m,x in zip(self.local,(a,g))],-1))
        return torch.cat([f(torch.cat((x,h(part)),-1)) for f,h,x,part in
                          zip(self.fuse,self.fold,(a,g),z.chunk(2,-1))],-1)


class MobileViT(nn.Module):
    def __init__(self,inc,width,count):
        super().__init__()
        self.local=nn.Sequential(Conv(inc,width,swish=True),Conv(width,width,1,swish=True))
        self.blocks=nn.Sequential(*[Block(width) for _ in range(count)])
        self.fold=Conv(width,inc,1,swish=True)
        self.fuse=Conv(inc*2,width,swish=True)

    def forward(self,x):
        return self.fuse(torch.cat((x,self.fold(self.blocks(self.local(x)))),-1))


class MobileHART(nn.Module):
    def __init__(self,classes,dropout=.3):
        super().__init__()
        def stem():
            return nn.Sequential(Conv(3,8,stride=2,swish=True), Inverted(8,32,16),
                Inverted(16,64,24,2),Inverted(24,96,24),Inverted(24,96,24),Inverted(24,96,32,2))
        self.acc,self.gyro=stem(),stem()
        self.sensor=SensorFusion()
        self.dense=linear(96,96)
        self.dropout=nn.Dropout(dropout)
        self.stage2=nn.Sequential(Inverted(96,384,80,2),MobileViT(80,120,4))
        self.stage3=nn.Sequential(Inverted(120,480,96,2),MobileViT(96,144,3))
        self.conv=Conv(144,384,1,swish=True)
        self.head,self.output=linear(384,1024),linear(1024,classes)

    def forward(self,x):
        x=self.sensor(self.acc(x[:,:,:3]),self.gyro(x[:,:,3:]))
        x=self.dropout(F.silu(self.dense(x)))
        x=self.conv(self.stage3(self.stage2(x))).mean(1)
        return self.output(self.dropout(F.silu(self.head(x))))
