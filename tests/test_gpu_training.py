"""Real CUDA smoke checks, skipped only on hosts without a GPU."""
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from autohar_researcher.harness import HARHarness
from autohar_researcher.types import Pipeline

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA GPU required')
@pytest.mark.parametrize('architecture',['bilstmattention','hart','mobilehart','tinyhar','tinierhar','deepconvlstm','spectra'])
def test_cuda_training_inference_and_augmentation(architecture,tmp_path,monkeypatch):
    task=SimpleNamespace(channels=list(range(6)),labels=['a','b'])
    harness=HARHarness(task,13,tmp_path,device='cuda')
    config=Pipeline.from_dict({'data':{'window_seconds':2.56},
        'model':{'architecture':architecture,'max_epochs':1,'batch_size':4},
        'augmentation':{'kind':'compose','transforms':[{'kind':'time_warp'},{'kind':'rotation'}],
                        'mixing':{'kind':'mixup'}}})
    original=harness._augment
    calls=[]
    def checked(p,x,rng,**kwargs):
        assert x.is_cuda and kwargs['labels'].is_cuda
        result=original(p,x,rng,**kwargs)
        assert result.is_cuda
        calls.append(result.device)
        return result
    monkeypatch.setattr(harness,'_augment',checked)
    x=np.random.default_rng(13).normal(size=(8,128,6)).astype(np.float32)
    model,epoch=harness._fit(config,x,np.arange(8)%2,epochs=1)
    assert all(p.is_cuda for p in model.parameters())
    assert epoch==1 and len(calls)==2
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert harness._predict(model,x,4).shape==(8,)
    assert 'tensorflow' not in sys.modules
