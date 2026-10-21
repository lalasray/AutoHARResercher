"""Pinned host loading and asynchronous device transfers for bounded GPU memory.

Files and fixed resampling are decoded on the host. DataLoader owns batching;
only batches reside on the GPU, so a complete LOSO corpus need not fit in VRAM.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def batches(x, y=None, *, batch_size, device, seed=13, shuffle=False, balanced=False):
    signals=torch.as_tensor(x,dtype=torch.float32)
    tensors=(signals,) if y is None else (signals,torch.as_tensor(y,dtype=torch.long))
    dataset=TensorDataset(*tensors)
    generator=torch.Generator().manual_seed(seed)
    sampler=None
    if balanced:
        labels=tensors[1]
        counts=torch.bincount(labels)
        sampler=WeightedRandomSampler(1/counts[labels].double(),len(labels),replacement=True,generator=generator)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle and sampler is None,
                      sampler=sampler,generator=generator,pin_memory=torch.device(device).type=='cuda',
                      num_workers=0,drop_last=False)
    for batch in loader:
        yield tuple(value.to(device,non_blocking=True) for value in batch)
