"""Device-resident PyTorch signal transforms and label mixing.

Uses the shared source-linked policy catalog. Random draws, interpolation, FFT,
spline solves and DTW cost matrices stay on the input device.
"""
import math
import torch
from torch.nn import functional as F
from .augmentations import validate_policy, validate_method, policy_steps, mixing_spec, validate_class_names


def resize(x,length):
    # x: T,C; align_corners reproduces the catalog's endpoint interpolation.
    return F.interpolate(x.T.unsqueeze(0),size=length,mode='linear',align_corners=True)[0].T


def interpolate(values,positions,query):
    idx=torch.searchsorted(positions.contiguous(),query.contiguous()).clamp(1,len(positions)-1)
    weight=(query-positions[idx-1])/(positions[idx]-positions[idx-1])
    return values[idx-1]+weight[:,None]*(values[idx]-values[idx-1])


def spline(values,length):
    """Uniform-knot not-a-knot cubic spline, matching SciPy CubicSpline."""
    n=values.shape[0]
    knots=torch.linspace(0,length-1,n,device=values.device,dtype=values.dtype)
    h=knots[1]-knots[0]
    if n==3:
        q=torch.arange(length,device=values.device,dtype=values.dtype)/h
        return (values[0]*(q-1)[:,None]*(q-2)[:,None]/2
                -values[1]*q[:,None]*(q-2)[:,None]
                +values[2]*q[:,None]*(q-1)[:,None]/2)
    matrix=torch.zeros(n,n,device=values.device,dtype=values.dtype)
    rhs=torch.zeros_like(values)
    matrix[0,:3]=torch.tensor([-1.,2.,-1.],device=values.device)
    matrix[-1,-3:]=torch.tensor([-1.,2.,-1.],device=values.device)
    ids=torch.arange(1,n-1,device=values.device)
    matrix[ids,ids-1]=1
    matrix[ids,ids]=4
    matrix[ids,ids+1]=1
    rhs[1:-1]=6*(values[2:]-2*values[1:-1]+values[:-2])/(h*h)
    second=torch.linalg.solve(matrix,rhs)
    t=torch.arange(length,device=values.device,dtype=values.dtype)
    i=(t/h).long().clamp(max=n-2)
    a=(knots[i+1]-t)/h
    b=1-a
    return a[:,None]*values[i]+b[:,None]*values[i+1]+(
        (a**3-a)[:,None]*second[i]+(b**3-b)[:,None]*second[i+1])*(h*h/6)


def dtw_path(a,b,ratio):
    n,m=len(a),len(b)
    band=max(abs(n-m),1,math.ceil(max(n,m)*ratio))
    cost=torch.full((n+1,m+1),float('inf'),device=a.device)
    cost[0,0]=0
    distance=torch.cdist(a,b,compute_mode='donot_use_mm_for_euclid_dist')
    for diagonal in range(2,n+m+1):
        i=torch.arange(max(1,diagonal-m),min(n,diagonal-1)+1,device=a.device)
        j=diagonal-i
        mask=(i-j).abs()<=band
        i,j=i[mask],j[mask]
        cost[i,j]=distance[i-1,j-1]+torch.minimum(cost[i-1,j-1],torch.minimum(cost[i-1,j],cost[i,j-1]))
    # Only the compact traceback control indices cross to CPU, never signals.
    directions=torch.stack((cost[:-1,:-1],cost[:-1,1:],cost[1:,:-1])).argmin(0).cpu()
    i,j=n,m
    path=[]
    while i and j:
        path.append((i-1,j-1))
        direction=int(directions[i-1,j-1])
        i-=direction!=2
        j-=direction!=1
    return torch.tensor(path[::-1],device=a.device,dtype=torch.long).T


def transform(x,name,p,generator,labels=None,normalizer=None):
    b,t,c=x.shape
    dev=x.device
    def rand(*shape): return torch.rand(shape,device=dev,generator=generator)
    def normal(*shape): return torch.randn(shape,device=dev,generator=generator)
    def integer(low,high): return int(torch.randint(low,high,(),device=dev,generator=generator))
    out=x.clone()
    if name=='jitter':
        return x+normal(b,t,c)*p['sigma']
    if name=='scaling':
        return x*(1+normal(b,1,c)*p['sigma'])
    if name=='rotation':
        if c%3:
            raise ValueError('Rotation requires complete consecutive XYZ triplets')
        axes=F.normalize(normal(b,3),dim=-1)
        angle=(rand(b)*2-1)*p['max_angle']
        skew=torch.zeros(b,3,3,device=dev)
        skew[:,0,1],skew[:,0,2]=-axes[:,2],axes[:,1]
        skew[:,1,0],skew[:,1,2]=axes[:,2],-axes[:,0]
        skew[:,2,0],skew[:,2,1]=-axes[:,1],axes[:,0]
        rotation=torch.eye(3,device=dev)[None]+angle.sin()[:,None,None]*skew+(1-angle.cos())[:,None,None]*(skew@skew)
        mean,scale=(0,1) if normalizer is None else (
            torch.as_tensor(normalizer.mean,device=dev,dtype=x.dtype),
            torch.as_tensor(normalizer.scale,device=dev,dtype=x.dtype))
        raw=x*scale+mean
        rotated=torch.einsum('bij,btgj->btgi',rotation,raw.reshape(b,t,c//3,3)).reshape_as(x)
        return (rotated-mean)/scale
    if name in ('magnitude_warp','time_warp'):
        for i in range(b):
            curve=spline(1+normal(p['knots']+2,c if name=='magnitude_warp' else 1)*p['sigma'],t)
            if name=='magnitude_warp':
                out[i]*=curve
            else:
                positions=curve[:,0].clamp_min(.001).cumsum(0)
                positions=(positions-positions[0])/(positions[-1]-positions[0])*(t-1)
                out[i]=interpolate(x[i],positions,torch.arange(t,device=dev,dtype=x.dtype))
    elif name=='permutation':
        for i in range(b):
            chunks=torch.tensor_split(torch.arange(t,device=dev),integer(1,min(t,p['max_segments'])+1))
            order=torch.randperm(len(chunks),device=dev,generator=generator).tolist()
            out[i]=x[i,torch.cat([chunks[j] for j in order])]
    elif name=='window_slice':
        size=max(2,min(t,math.ceil(t*p['ratio'])))
        for i in range(b):
            start=integer(0,t-size+1)
            out[i]=resize(x[i,start:start+size],t)
    elif name=='window_warp':
        size=max(2,min(t-2,math.ceil(t*p['ratio'])))
        for i in range(b):
            start=integer(1,t-size)
            scale=p['scales'][integer(0,len(p['scales']))]
            part=resize(x[i,start:start+size],max(2,round(size*scale)))
            out[i]=resize(torch.cat((x[i,:start],part,x[i,start+size:])),t)
    elif name in ('time_mask','frequency_mask'):
        z=torch.fft.rfft(x,dim=1) if name=='frequency_mask' else out
        lower=1 if name=='frequency_mask' else 0
        maximum=max(1,round((z.shape[1]-lower)*p['ratio']))
        for i in range(b):
            size=integer(0,maximum+1)
            start=integer(lower,z.shape[1]-size+1)
            z[i,start:start+size]=0
        out=torch.fft.irfft(z,n=t,dim=1) if lower else z
    elif name=='channel_dropout':
        mask=rand(b,c)<p['rate']
        keep=torch.randint(c,(b,),device=dev,generator=generator)
        mask[torch.arange(b,device=dev),keep]&=~mask.all(1)
        out.masked_fill_(mask[:,None,:],0)
    elif name in ('spawner','random_guided_warp'):
        if labels is None:
            raise ValueError('DTW augmentation requires hard training labels')
        for i in range(b):
            candidates=torch.where((labels==labels[i])&(torch.arange(b,device=dev)!=i))[0]
            if not len(candidates):
                continue
            donor=x[candidates[integer(0,len(candidates))]]
            if name=='random_guided_warp':
                path=dtw_path(donor,x[i],p['band_ratio'])
                out[i]=resize(x[i,path[1]],t)
            else:
                split=integer(1,t-1)
                path=torch.cat((dtw_path(x[i,:split],donor[:split],p['band_ratio']),
                                dtw_path(x[i,split:],donor[split:],p['band_ratio'])+split),1)
                out[i]=resize((x[i,path[0]]+donor[path[1]])/2,t)+normal(t,c)*p['sigma']
    else:
        raise ValueError(f'Unknown transform: {name}')
    return out


def policy_groups(policy, labels, class_names):
    if class_names is None or labels is None:
        raise ValueError('Class-specific augmentation requires hard labels and the fixed class names')
    validate_class_names(policy, class_names)
    if labels.ndim != 1 or labels.dtype != torch.long or ((labels < 0) | (labels >= len(class_names))).any():
        raise ValueError('Class-specific augmentation requires valid hard class indices')
    remaining = torch.ones_like(labels, dtype=torch.bool)
    for name in sorted(policy['per_class']):
        override = policy['per_class'][name]
        mask = labels == class_names.index(name)
        remaining &= ~mask
        if mask.any():
            yield mask, override
    if remaining.any():
        yield remaining, {k: v for k, v in policy.items() if k != 'per_class'}


def augment(x,policy,generator,labels=None,normalizer=None,class_names=None):
    validate_policy(policy)
    if policy.get('per_class'):
        out = x.clone()
        for mask, resolved in policy_groups(policy, labels, class_names):
            # Full batch keeps same-class DTW donors available; only the
            # recipient rows of this policy are written to the result.
            result = augment(x, resolved, generator, labels, normalizer)
            out[mask] = result[mask]
        return out
    if x.ndim!=3 or x.shape[1]<4 or not torch.isfinite(x).all():
        raise ValueError('Expected finite [batch,time>=4,channels] signals')
    out=x.clone()
    for step in policy_steps(policy):
        probability=step.get('probability',1)*(policy.get('probability',1) if policy.get('kind')=='compose' else 1)
        mask=torch.rand(len(x),device=x.device,generator=generator)<probability
        if mask.any():
            result=transform(out,step['kind'],validate_method(step),generator,labels,normalizer)
            out=torch.where(mask[:,None,None],result,out)
    if not torch.isfinite(out).all():
        raise ValueError('Non-finite augmentation')
    return out


def mix_batch(x,labels,policy,generator,num_classes,class_names=None):
    validate_policy(policy)
    if policy.get('per_class'):
        if class_names is None or len(class_names) != num_classes:
            raise ValueError('Class names must match the fixed output dimensions')
        groups = list(policy_groups(policy, labels, class_names))
        if not any(mixing_spec(resolved) is not None for _, resolved in groups):
            return x, labels
        out = x.clone()
        targets = F.one_hot(labels, num_classes).to(x.dtype)
        for mask, resolved in groups:
            mixed, soft = mix_batch(x, labels, resolved, generator, num_classes)
            out[mask] = mixed[mask]
            targets[mask] = soft[mask] if soft.ndim == 2 else F.one_hot(soft[mask], num_classes).to(x.dtype)
        return out, targets
    spec=mixing_spec(policy)
    if spec is None:
        return x,labels
    p=validate_method(spec,mixing_only=True)
    b,t,_=x.shape
    targets=F.one_hot(labels,num_classes).to(x.dtype)
    if b<2:
        return x.clone(),targets
    order=torch.randperm(b,device=x.device,generator=generator)
    donors=torch.empty_like(order)
    donors[order]=order.roll(1)
    concentration=torch.full((b,),p['alpha'],device=x.device,dtype=x.dtype)
    a=torch._standard_gamma(concentration,generator=generator)
    c=torch._standard_gamma(concentration,generator=generator)
    weight=(a/(a+c).clamp_min(torch.finfo(x.dtype).tiny)).clamp(0,1)
    active=torch.rand(b,device=x.device,generator=generator)<spec.get('probability',1)
    weight=torch.where(active,weight,torch.ones_like(weight))
    if spec['kind']=='mixup':
        out=weight[:,None,None]*x+(1-weight[:,None,None])*x[donors]
    else:
        size=((1-weight)*t).round().long()
        start=(torch.rand(b,device=x.device,generator=generator)*(t-size+1)).long()
        time=torch.arange(t,device=x.device)[None]
        mask=(time>=start[:,None])&(time<(start+size)[:,None])
        out=torch.where(mask[:,:,None],x[donors],x)
        weight=1-size.to(x.dtype)/t
    targets=weight[:,None]*targets+(1-weight[:,None])*targets[donors]
    if not torch.isfinite(out).all() or not torch.isfinite(targets).all():
        raise ValueError('Non-finite mixed batch')
    return out,targets
