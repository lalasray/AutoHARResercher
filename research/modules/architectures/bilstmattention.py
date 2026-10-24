"""PyTorch port of the author's CNN–ResBLSTM–Attention architecture."""
import torch
from torch import nn
from .torch_author import BatchNorm, Conv, linear


class Branch(nn.Module):
    def __init__(self,steps,kernel,pool,dropout):
        super().__init__()
        self.convs=nn.ModuleList([Conv(a,b,kernel,swish=True) for a,b in zip([3,256,128,64],[256,128,64,32])])
        self.norms=nn.ModuleList([BatchNorm(c) for c in [256,128,64,32]])
        self.pool=nn.MaxPool1d(pool)
        self.dropout=nn.Dropout(dropout)
        self.norm1,self.norm2=nn.LayerNorm(32,eps=.001),nn.LayerNorm(128,eps=.001)
        self.lstm1=nn.LSTM(32,64,batch_first=True,bidirectional=True)
        self.lstm2=nn.LSTM(128,64,batch_first=True,bidirectional=True)
        for lstm in (self.lstm1,self.lstm2):
            for suffix in ('','_reverse'):
                nn.init.xavier_uniform_(getattr(lstm,'weight_ih_l0'+suffix))
                nn.init.orthogonal_(getattr(lstm,'weight_hh_l0'+suffix))
                nn.init.zeros_(getattr(lstm,'bias_ih_l0'+suffix))
                nn.init.zeros_(getattr(lstm,'bias_hh_l0'+suffix))
                with torch.no_grad():
                    getattr(lstm,'bias_ih_l0'+suffix)[64:128].fill_(1)
                # Keras has a single bias; avoid doubling its update in PyTorch.
                getattr(lstm,'bias_hh_l0'+suffix).requires_grad_(False)
        self.attention_weight=nn.Parameter(torch.empty(128))
        nn.init.uniform_(self.attention_weight,-(3/128)**.5,(3/128)**.5)
        self.attention_bias=nn.Parameter(torch.zeros(steps//pool**4))

    def forward(self,x):
        for conv,norm in zip(self.convs,self.norms):
            x=self.dropout(self.pool(norm(conv(x)).transpose(1,2)).transpose(1,2))
        x=self.lstm1(self.norm1(x))[0]
        x=x+self.lstm2(self.norm2(x))[0]
        a=torch.exp(torch.tanh(x@self.attention_weight+self.attention_bias))
        a=a/(a.sum(1,keepdim=True)+1e-7)
        return (x*a.unsqueeze(-1)).sum(1)


class BiLSTMAttention(nn.Module):
    def __init__(self,steps,channels,classes,params):
        super().__init__()
        self.branches=nn.ModuleList([Branch(steps,params.get('kernel_size',3),
            params.get('pool_size',2),params.get('dropout',.2)) for _ in range(channels//3)])
        self.dropout=nn.Dropout(params.get('head_dropout',.5))
        self.output=linear(128*len(self.branches),classes)
        self.norm=BatchNorm(classes)

    def forward(self,x):
        x=torch.cat([branch(x[:,:,i*3:i*3+3]) for i,branch in enumerate(self.branches)],-1)
        return self.norm(self.output(self.dropout(x)))


def build_model(input_dim,num_classes,params):
    channels=params['input_channels']
    steps=input_dim//channels
    fixed={'hidden_dim':64,'conv_channels':[256,128,64,32],'branch_channels':3,
           'classifier_batch_norm':True,'norm_eps':.001,'batch_norm_momentum':.99}
    if channels%3:
        raise ValueError('Author BiLSTM-Attention requires complete sensor triplets')
    for key,value in fixed.items():
        if params.get(key,value)!=value:
            raise ValueError(f'Author BiLSTM-Attention fixes {key}={value}')
    if steps//params.get('pool_size',2)**4<1:
        raise ValueError('Window too short for the author pooling stack')
    return BiLSTMAttention(steps,channels,num_classes,params)
