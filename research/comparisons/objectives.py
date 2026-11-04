import numpy as np
import torch
from torch.nn import functional as F


def contrastive(a, b, temperature, labels=None):
    z = F.normalize(torch.cat([a,b]), dim=1)
    n = len(a)
    if n < 2:
        raise ValueError('Contrastive batches require at least two examples')
    logits = z @ z.T / temperature
    eye = torch.eye(2*n, device=z.device, dtype=torch.bool)
    log_probability = logits - torch.logsumexp(logits.masked_fill(eye, -torch.inf), dim=1, keepdim=True)
    if labels is None:
        positives = torch.zeros_like(eye)
        index = torch.arange(n, device=z.device)
        positives[index, index+n] = True
        positives[index+n, index] = True
    else:
        repeated = torch.cat([labels,labels])
        positives = repeated[:,None].eq(repeated[None]) & ~eye
    return -(log_probability.masked_fill(~positives,0).sum(1) / positives.sum(1)).mean()


def views(x, rng, options):

    factor = torch.as_tensor(rng.normal(2., options['scaling_sigma'], (len(x),x.shape[1],1)), device=x.device,dtype=x.dtype)
    weak = x * factor
    strong = x.clone()
    for i in range(len(x)):
        count = int(rng.integers(1,min(options['max_segments'], x.shape[1])+1))
        cuts = sorted(rng.choice(np.arange(1,x.shape[1]),count-1,replace=False))
        chunks = np.split(np.arange(x.shape[1]),cuts)
        indices = np.concatenate([chunks[k] for k in rng.permutation(count)])
        strong[i] = x[i,torch.as_tensor(indices,device=x.device)]
    noise = torch.as_tensor(rng.normal(0,options['jitter_sigma'],x.shape),device=x.device,dtype=x.dtype)
    return weak, strong+noise


def masked_view(x, rng, options):
    mask = torch.zeros(x.shape[:2],device=x.device,dtype=torch.bool)
    wanted = max(1,round(x.shape[1]*options['mask_ratio']))
    for row in range(len(x)):
        selected = set()
        while len(selected) < wanted:
            start = int(rng.integers(x.shape[1]))
            span = int(rng.integers(1,options['max_span']+1))
            for index in range(start,min(start+span,x.shape[1])):
                selected.add(index)
                if len(selected) == wanted:
                    break
        mask[row,list(sorted(selected))] = True
    result = x.clone()
    for row in range(len(x)):
        draw = rng.random()
        if draw < .8:
            result[row,mask[row]] = 0
        elif draw < .9:
            result[row,mask[row]] = torch.as_tensor(rng.random((int(mask[row].sum()),x.shape[-1])),device=x.device,dtype=x.dtype)
    return result, mask


def semantic_weights(lam, radius_a, radius_b):
    a, b = lam*radius_a, (1-lam)*radius_b
    total = a+b

    return torch.where(total > 1e-12, a/total.clamp_min(1e-12), torch.full_like(total,lam)).detach()


def large_margin(logits, labels, features, gamma, top_k=1):
    probabilities = logits.softmax(1)
    correct = probabilities.gather(1,labels[:,None])
    others = probabilities.masked_fill(F.one_hot(labels,logits.shape[1]).bool(),-torch.inf)
    differences = correct - others.topk(min(top_k,logits.shape[1]-1),dim=1).values
    gradients = [torch.autograd.grad(differences[:,k].sum(),features,retain_graph=True)[0]
                 for k in range(differences.shape[1])]
    norms = torch.stack([g.flatten(1).norm(dim=1) for g in gradients],dim=1).detach()
    distances = (differences / (norms+1e-8)).mean(1)

    clipped = distances.clamp_min(-3*gamma)
    return F.relu(gamma-clipped)-gamma
