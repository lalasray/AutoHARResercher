from copy import deepcopy
import itertools
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from .settings import settings
from .objectives import contrastive, large_margin, masked_view, semantic_weights, views


def minibatches(x, y, size, rng, device, indices=None):
    indices = rng.permutation(len(x)) if indices is None else rng.permutation(indices)
    if len(indices) < 2 or size < 2:
        raise ValueError('Comparison training requires at least two samples and batch_size >= 2')
    groups = list(np.array_split(indices, max(1,int(np.ceil(len(indices)/size)))))
    if len(groups[-1]) == 1 and len(groups) > 1:
        groups[-2] = np.concatenate([groups[-2],groups[-1]])
        groups.pop()
    for index in groups:
        yield (torch.as_tensor(x[index],device=device,dtype=torch.float32),
               torch.as_tensor(y[index],device=device,dtype=torch.long),index)


def label_subset(y, fraction, rng):
    selected = []
    for label in np.unique(y):
        members = np.flatnonzero(y==label)
        selected.extend(rng.permutation(members)[:max(1,int(np.ceil(len(members)*fraction)))])
    return np.asarray(selected,dtype=int)


def set_trainable(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def source_radii(model, x, y, domains, options, device, batch_size):
    model.eval()

    sums, counts = {}, {}
    with torch.no_grad():
        for start in range(0,len(x),batch_size):
            feature = model.features(torch.as_tensor(x[start:start+batch_size],device=device,dtype=torch.float32))
            for row in range(len(feature)):
                key = (str(domains[start+row]),int(y[start+row]))
                sums[key] = sums.get(key,torch.zeros_like(feature[row]))+feature[row]
                counts[key] = counts.get(key,0)+1
        means = {key:value/counts[key] for key,value in sums.items()}
        distances = {key:torch.zeros((),device=device) for key in means}
        for start in range(0,len(x),batch_size):
            feature = model.features(torch.as_tensor(x[start:start+batch_size],device=device,dtype=torch.float32))
            for row in range(len(feature)):
                key = (str(domains[start+row]),int(y[start+row]))
                distance = (feature[row]-means[key]).norm(p=options['radius_norm'])
                distances[key] = (torch.maximum(distances[key],distance) if options['radius']=='max'
                                  else distances[key]+distance/counts[key])
    return distances


def fit(harness, p, x, y, validation=None, epochs=None):
    from autohar_researcher.harness import domain_scores
    if harness.task.protocol != 'lodo' or len(harness.task.channels) != 6:
        raise ValueError('Comparison methods require six-channel LODO tasks')
    method = p.model['architecture']
    options = settings(method,p.model.get('custom_params',{}))
    harness._seed()
    model = harness._build_model(p)
    device, size = harness.device, p.model['batch_size']
    rng = np.random.default_rng(harness.seed)
    started = time.monotonic()
    phase = 'search' if validation is not None else 'final_refit'
    records = []

    def record(stage, epoch, loss=None, score=None, samples=None):
        row = dict(method=method,phase=phase,stage=stage,epoch=epoch,loss=loss,
                   validation_macro_f1=score,samples=samples,elapsed_seconds=time.monotonic()-started)
        with (harness.output/'comparison-training.jsonl').open('a') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\n')
        records.append(row)

    def optimizer():
        return harness._optimizer(model,p)

    def update(opt, loss):
        if loss.ndim or not torch.isfinite(loss):
            raise ValueError('Nonfinite or nonscalar comparison objective')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.detach())

    def contrast_stage(stage, n_epochs, targets, supervised=False):
        opt = optimizer()
        for epoch in range(1,n_epochs+1):
            model.train()
            losses=[]
            for batch, labels, _ in minibatches(x,targets,size,rng,device):
                a,b = views(batch,rng,options)
                fa,fb = model.features(a),model.features(b)
                t1,z1 = model.temporal(fa,fb)
                t2,z2 = model.temporal(fb,fa)
                contextual = contrastive(z1,z2, .07 if supervised else options['temperature'],labels if supervised else None)
                loss = .01*(t1+t2)+.1*contextual if supervised else t1+t2+.7*contextual
                losses.append(update(opt,loss))
            record(stage,epoch,float(np.mean(losses)),samples=len(x))

    supervised_indices = np.arange(len(x))
    if method == 'contratsc':
        contrast_stage('temporal_contextual_pretraining',options['pretrain_epochs'],y)
        supervised_indices = label_subset(y,options['label_fraction'],rng)
        model.head.reset_parameters()
        opt = optimizer()
        for epoch in range(1,options['teacher_epochs']+1):
            model.train()
            losses=[]
            for batch,labels,_ in minibatches(x,y,size,rng,device,supervised_indices):
                losses.append(update(opt,F.cross_entropy(model(batch),labels)))
            record('teacher_finetuning',epoch,float(np.mean(losses)),samples=len(supervised_indices))
        pseudo = harness._predict(model,x,size)
        record('source_pseudo_labels',0,samples=len(pseudo))

        model.context = type(model.context)(128).to(device)
        for layer in model.predictors:
            layer.reset_parameters()
        for layer in model.projector:
            if hasattr(layer,'reset_parameters'):
                layer.reset_parameters()
        contrast_stage('class_aware_pretraining',options['class_aware_epochs'],pseudo,True)
        set_trainable(model,False)
        set_trainable(model.head,True)
        model.head.reset_parameters()
    elif method == 'crosshar':
        opt = optimizer()
        permutations = list(itertools.permutations(range(3)))
        for stage, n_epochs in [('masked_reconstruction',options['reconstruction_epochs']),
                                ('hierarchical_contrastive',options['contrastive_epochs'])]:
            for epoch in range(1,n_epochs+1):
                model.train()
                losses=[]

                for permutation in permutations:
                    columns = list(permutation)+[i+3 for i in permutation]
                    for batch,_,_ in minibatches(x,y,size,rng,device):
                        a,b = views(batch[:,:,columns],rng,options)
                        a,b = model.normalize(a),model.normalize(b)
                        ma,ia = masked_view(a,rng,options)
                        mb,ib = masked_view(b,rng,options)
                        za,zb = model.features(ma),model.features(mb)
                        reconstruction = (F.mse_loss(model.decoder(za)[ia],a[ia])+F.mse_loss(model.decoder(zb)[ib],b[ib]))/2
                        loss = reconstruction
                        if stage == 'hierarchical_contrastive':
                            ca,cb = model.projector(model.context(za)),model.projector(model.context(zb))
                            loss = options['reconstruction_weight']*reconstruction+contrastive(ca,cb,options['temperature'])
                        losses.append(update(opt,loss))
                record(stage,epoch,float(np.mean(losses)),samples=6*len(x))
        supervised_indices = label_subset(y,options['label_fraction'],rng)
        set_trainable(model,False)
        for module in [model.class_project,model.classifier,model.head]:
            set_trainable(module,True)
    domains = getattr(harness,'_comparison_domains',None)
    if method == 'sdmix' and (domains is None or len(domains)!=len(x) or len(set(domains))<2):
        raise ValueError('SDMix requires windows from at least two source domains')
    opt = optimizer()
    best, best_epoch, state, stale = -float('inf'),0,None,0
    harness._last_lr_schedule=[]
    radii = None
    for epoch in range(1,(epochs or p.model['max_epochs'])+1):
        losses=[]
        if method == 'sdmix':
            if radii is None or (epoch-1)%options['radius_update_epochs']==0:
                radii = source_radii(model,x,y,domains,options,device,size)
            model.train()
            unique = sorted(set(domains))

            steps=max(int(np.ceil(np.sum(domains==domain)/size)) for domain in unique)
            for _ in range(steps):
                order = rng.permutation(unique)
                loss=0
                for i,domain in enumerate(order):
                    other=order[(i+1)%len(order)]
                    ia=rng.choice(np.flatnonzero(domains==domain),size,replace=True)
                    ib=rng.choice(np.flatnonzero(domains==other),size,replace=True)
                    a=torch.as_tensor(x[ia],device=device,dtype=torch.float32)
                    b=torch.as_tensor(x[ib],device=device,dtype=torch.float32)
                    ya=torch.as_tensor(y[ia],device=device,dtype=torch.long)
                    yb=torch.as_tensor(y[ib],device=device,dtype=torch.long)
                    lam=float(rng.beta(options['mixup_alpha'],options['mixup_alpha']))
                    r1=torch.stack([radii[(str(domain),int(v))] for v in y[ia]])
                    r2=torch.stack([radii[(str(other),int(v))] for v in y[ib]])
                    weight=semantic_weights(lam,r1,r2)
                    feature=model.features(lam*a+(1-lam)*b)
                    logits=model.head(feature)
                    la=large_margin(logits,ya,feature,options['margin'],options['top_k'])
                    lb=large_margin(logits,yb,feature,options['margin'],options['top_k'])
                    loss=loss+(weight*la+(1-weight)*lb).mean()/len(unique)
                losses.append(update(opt,loss))
        else:

            model.eval()
            if method=='crosshar':
                model.classifier.train()
            model.head.train()
            for batch,labels,_ in minibatches(x,y,size,rng,device,supervised_indices):
                losses.append(update(opt,F.cross_entropy(model(batch),labels)))
        harness._last_lr_schedule.append(opt.param_groups[0]['lr'])
        score=None
        if validation is not None:
            vx,vy,vd=validation
            scores=domain_scores(harness._predict(model,vx,size),vy,vd,len(harness.task.labels))
            score=float(np.mean(list(scores.values())))
            if score>best:
                best,best_epoch,state,stale=score,epoch,deepcopy(model.state_dict()),0
            else:
                stale+=1
        else:
            best_epoch=epoch
        record('semantic_discriminative_training' if method=='sdmix' else 'supervised_classifier',
               epoch,float(np.mean(losses)),score,len(supervised_indices))
        if validation is not None and stale>=p.model['patience']:
            break
    if state is not None:
        model.load_state_dict(state)
    report=dict(method=method,phase=phase,settings=options,selected_classifier_epochs=best_epoch,
                training_windows=len(x),labeled_windows=len(supervised_indices),stages=records,
                scope='source training only; final refit uses combined non-test source train+validation')
    (harness.output/f'comparison-{phase}.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return model,best_epoch
