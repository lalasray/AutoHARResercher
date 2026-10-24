import json
import pytest
from autohar_researcher.agents import Search
from autohar_researcher.types import Pipeline
from test_search import Harness, Agents


def test_extended_budget_requires_exploration(tmp_path):
    with pytest.raises(ValueError, match='Paper limits'):
        Search(Harness(), Agents(), tmp_path/'standard',rounds=1000,budget=3000)
    s=Search(Harness(), Agents(), tmp_path/'explore',rounds=1000,budget=3000,
             exploration=True,target_improvements=10)
    assert s.target_improvements==10


def test_native_100hz_windows_preserve_samples(tmp_path):
    import numpy as np
    from autohar_researcher.datasets import Recording, windows
    x=np.arange(6000,dtype=np.float32).reshape(1000,6)
    path=tmp_path/'native.npy';np.save(path,x)
    r=Recording('wisdm','1','1',path,'walking',100,('ax','ay','az','gx','gy','gz'),(1.,)*6)
    native,_,_=windows([r],('walking',),4,.5,sampling_hz=100)
    assert native.shape==(4,400,6)
    np.testing.assert_array_equal(native[0],x[:400])
    np.testing.assert_array_equal(native[1],x[200:600])
    conventional,_,_=windows([r],('walking',),4,.5)
    assert conventional.shape==(4,200,6)


def test_harness_propagates_manifest_rate(manifest,tmp_path):
    import torch
    from autohar_researcher.datasets import make_task
    from autohar_researcher.harness import HARHarness
    raw=json.loads(manifest.read_text());raw['processing_hz']=100
    manifest.write_text(json.dumps(raw))
    task=make_task(manifest,'loso','uci_har','0',13)
    assert task.sampling_hz==100 and task.public_metadata()['sampling_hz']==100
    p=Pipeline.from_dict({'data':{'window_seconds':.5},'model':{'architecture':'cnn'}})
    channels=len(task.channels)
    p.code['research/components/model.py']=f'''import torch
def build_model(input_dim,num_classes,params):
    assert input_dim == {50*channels}
    return torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(input_dim,num_classes))
'''
    h=HARHarness(task,13,tmp_path/'fit','cpu')
    h.preflight(p)
    x,_,_=h._training_arrays(p,fit=False)
    vx,_,_=h._validation_arrays(p)
    assert x.shape[1]==vx.shape[1]==50
    assert h._build_model(p)(torch.zeros(2,50,channels)).shape==(2,len(task.labels))


def test_native_20hz_windows(tmp_path):
    import numpy as np
    from autohar_researcher.datasets import Recording,windows
    x=np.arange(600,dtype=np.float32).reshape(200,3);path=tmp_path/'x.npy';np.save(path,x)
    r=Recording('wisdm','1','1',path,'walking',20,('ax','ay','az'),(1.,)*3)
    wx,_,_=windows([r],('walking',),4,.5,sampling_hz=20)
    assert wx.shape==(4,80,3)
    np.testing.assert_array_equal(wx[1],x[40:120])
