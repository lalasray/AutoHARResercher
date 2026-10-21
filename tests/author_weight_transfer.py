"""Test-only weight transfer from pinned author TensorFlow graphs to torch ports."""
import re
import numpy as np
import torch
from research.modules.architectures.torch_author import Conv, BatchNorm


def put(target, value):
    with torch.no_grad():
        target.copy_(torch.as_tensor(np.asarray(value),dtype=target.dtype))


def dense(target, source):
    w,b=source.get_weights()
    put(target.weight,w.T); put(target.bias,b)


def norm(target,source):
    weights=source.get_weights()
    put(target.weight,weights[0]);put(target.bias,weights[1])
    if isinstance(target,BatchNorm):
        put(target.running_mean,weights[2]);put(target.running_var,weights[3])


def conv(target,source):
    w,*bias=source.get_weights()
    w=w.transpose(1,2,0) if source.__class__.__name__=='DepthwiseConv1D' else w.transpose(2,1,0)
    put(target.op.weight,w)
    if bias: put(target.op.bias,bias[0])


def mha(target,source):
    weights=source.get_weights()
    for i,layer in enumerate((target.q,target.k,target.v)):
        w,b=weights[2*i:2*i+2]
        put(layer.weight,w.reshape(w.shape[0],-1).T);put(layer.bias,b.reshape(-1))
    w,b=weights[-2:]
    put(target.out.weight,w.reshape(-1,w.shape[-1]).T);put(target.out.bias,b)


def layers(reference,kind):
    selected=[l for l in reference.layers if l.__class__.__name__==kind]
    def order(layer):
        match=re.search(r'_(\d+)$',layer.name)
        return int(match.group(1))+1 if match else 0
    return sorted(selected,key=order)


def sensor(block,reference,index):
    lite=reference.get_layer('liteFormer_'+str(index))
    put(block.lite.weight,np.stack([w.numpy().reshape(-1) for w in lite.depthwise_kernel])[:,None])
    mha(block.acc,reference.get_layer('AccMHA_'+str(index)).MHA)
    mha(block.gyro,reference.get_layer('GyroMHA_'+str(index)).MHA)
    norm(block.norm1,reference.get_layer('normalizedInputs_'+str(index)))


def transfer_hart(model,reference):
    patch=next(l for l in reference.layers if l.__class__.__name__=='SensorPatches')
    conv(model.acc,patch.accProjection);conv(model.gyro,patch.gyroProjection)
    position=next(l for l in reference.layers if l.__class__.__name__=='PatchEncoder')
    put(model.position,position.get_weights()[0])
    for i,block in enumerate(model.blocks): sensor(block,reference,i)
    norms=[b.norm2 for b in model.blocks]+[model.norm]
    ordinary=[l for l in layers(reference,'LayerNormalization') if not l.name.startswith('normalizedInputs_')]
    for a,b in zip(norms,ordinary,strict=True): norm(a,b)
    targets=[v for b in model.blocks for v in (b.fc1,b.fc2)]+[model.head,model.output]
    for a,b in zip(targets,layers(reference,'Dense'),strict=True): dense(a,b)


def transfer_mobile(model,reference):
    # Follow the author's creation order, which differs from graph execution order.
    convs=[model.acc[0],model.gyro[0]]
    inverted=list(model.acc[1:])+list(model.gyro[1:])
    for inv in inverted: convs.extend([inv.expand,inv.project])
    for local in model.sensor.local: convs.extend(local)
    for fold,fuse in zip(model.sensor.fold,model.sensor.fuse): convs.extend([fold,fuse])
    for stage in (model.stage2,model.stage3):
        inv,vit=stage
        inverted.append(inv)
        convs.extend([inv.expand,inv.project,*vit.local,vit.fold,vit.fuse])
    convs.append(model.conv)
    for a,b in zip(convs,layers(reference,'Conv1D'),strict=True): conv(a,b)
    for a,b in zip(inverted,layers(reference,'DepthwiseConv1D'),strict=True): conv(a.depth,b)
    norms=[n for inv in inverted for n in (inv.bn1,inv.bn2,inv.bn3)]
    for a,b in zip(norms,layers(reference,'BatchNormalization'),strict=True): norm(a,b)
    for i,block in enumerate(model.sensor.blocks): sensor(block,reference,i)
    global_blocks=list(model.stage2[1].blocks)+list(model.stage3[1].blocks)
    for a,b in zip(global_blocks,layers(reference,'MultiHeadAttention'),strict=True): mha(a.attention,b)
    normtargets=[b.norm2 for b in model.sensor.blocks]+[n for b in global_blocks for n in (b.norm1,b.norm2)]
    ordinary=[l for l in layers(reference,'LayerNormalization') if not l.name.startswith('normalizedInputs_')]
    for a,b in zip(normtargets,ordinary,strict=True): norm(a,b)
    targets=[n for b in model.sensor.blocks for n in (b.fc1,b.fc2)]+[model.dense]
    targets += [n for b in global_blocks for n in (b.fc1,b.fc2)]+[model.head,model.output]
    for a,b in zip(targets,layers(reference,'Dense'),strict=True): dense(a,b)


def transfer_bilstm(model,reference):
    for a,b in zip([c for branch in model.branches for c in branch.convs],layers(reference,'Conv1D'),strict=True): conv(a,b)
    for a,b in zip([c for branch in model.branches for c in branch.norms]+[model.norm],layers(reference,'BatchNormalization'),strict=True): norm(a,b)
    for a,b in zip([c for branch in model.branches for c in (branch.norm1,branch.norm2)],layers(reference,'LayerNormalization'),strict=True): norm(a,b)
    for a,b in zip([c for branch in model.branches for c in (branch.lstm1,branch.lstm2)],layers(reference,'Bidirectional'),strict=True):
        for suffix,layer in [('',b.forward_layer),('_reverse',b.backward_layer)]:
            w,r,bias=layer.get_weights()
            put(getattr(a,'weight_ih_l0'+suffix),w.T)
            put(getattr(a,'weight_hh_l0'+suffix),r.T)
            put(getattr(a,'bias_ih_l0'+suffix),bias)
            put(getattr(a,'bias_hh_l0'+suffix),np.zeros_like(bias))
    for a,b in zip(model.branches,layers(reference,'Attention'),strict=True):
        w,bias=b.get_weights()
        put(a.attention_weight,w);put(a.attention_bias,bias)
    dense(model.output,layers(reference,'Dense')[0])
