import json

import numpy as np
import pytest
import torch

from autohar_researcher.datasets import make_task
from autohar_researcher.harness import HARHarness
from autohar_researcher.patches import validate_pipeline
from autohar_researcher.types import Pipeline
from research.comparisons.objectives import contrastive, large_margin, semantic_weights, masked_view
from research.comparisons.settings import DEFAULTS


def small_pipeline(method):
    options = dict(DEFAULTS[method])
    for key in options:
        if key.endswith('epochs'):
            options[key] = 1
    return Pipeline.from_dict({'data': {'window_seconds':1,'overlap':0},
        'model': {'architecture':method,'max_epochs':2,'patience':1,'batch_size':32,'custom_params':options}})


def test_domain_labels_follow_custom_sampling_and_synthetic_samples(manifest, tmp_path):
    from types import SimpleNamespace
    from autohar_researcher.datasets import windows
    task = make_task(manifest, 'lodo', 'uci_har')
    generator = SimpleNamespace(generate=lambda request, length, channels, *args, **kwargs:
                                np.zeros((2, length, channels), dtype=np.float32))
    h = HARHarness(task, 13, tmp_path, device='cpu', generator=generator)
    p = small_pipeline('sdmix')
    p.augmentation['synthetic'] = [{'generator': 'text', 'label': task.labels[0],
        'count': 2, 'ratio': 1.0, 'prompt': 'walking', 'source_dataset': 'motion_sense'}]
    p.code['research/components/data.py'] = '''import numpy as np
def sample_indices(x, y, rng):
    return np.tile(np.arange(len(y) - 1, -1, -1), 30)
'''
    _, raw_y, domains = windows(task.train, task.labels, 1, 0)
    rng = np.random.default_rng(13)
    selected = np.concatenate([rng.permutation(np.flatnonzero(raw_y == i))
                               for i in range(len(task.labels))])
    expected_domains = np.tile(domains[selected][::-1], 30)
    x, y, _ = h._training_arrays(p, False)
    assert len(h._comparison_domains) == len(x) == len(y)
    np.testing.assert_array_equal(h._comparison_domains[:-2], expected_domains)
    assert h._comparison_domains[-2:].tolist() == ['motion_sense', 'motion_sense']


@pytest.mark.parametrize('method', ['contratsc','sdmix','crosshar'])
def test_complete_method_source_isolation_and_final_retraining(method,manifest,tmp_path,monkeypatch):
    torch.set_num_threads(1)
    task=make_task(manifest,'lodo','uci_har')
    h=HARHarness(task,13,tmp_path/method,device='cpu')
    p=small_pipeline(method)
    held_out={r.path for r in task.test}
    validation={r.path for r in task.validation}
    accessed=[]
    original=np.load
    def load(path,*args,**kwargs):
        accessed.append(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',load)

    x,y,norm=h._training_arrays(p,False)
    assert not set(accessed)&(held_out|validation)
    if method=='sdmix':
        assert set(h._comparison_domains)=={'motion_sense','real_life_har'}
    result=h.evaluate(p)
    assert np.isfinite(result.score)
    assert not set(accessed)&held_out
    report=json.loads((h.output/'comparison-search.json').read_text())
    stages={r['stage'] for r in report['stages']}
    expected={'contratsc':{'temporal_contextual_pretraining','teacher_finetuning','source_pseudo_labels','class_aware_pretraining','supervised_classifier'},
        'crosshar':{'masked_reconstruction','hierarchical_contrastive','supervised_classifier'},
        'sdmix':{'semantic_discriminative_training'}}[method]
    assert stages==expected
    final=h.finalize(p)
    replay=json.loads((h.output/'comparison-final_refit.json').read_text())
    assert replay['training_windows']>report['training_windows']
    assert replay['selected_classifier_epochs']==result.best_epoch==final['refit_epochs']
    assert {r['stage'] for r in replay['stages']}==expected
    assert all(r['validation_macro_f1'] is None for r in replay['stages'])
    assert set(accessed)&held_out==held_out
    checkpoint=torch.load(h.output/'final-model.pt',weights_only=False)
    restored=h._build_model(p)
    restored.load_state_dict(checkpoint['state_dict'])
    assert restored(torch.zeros(2,50,6)).shape==(2,5)
    with pytest.raises(RuntimeError,match='once'):
        h.finalize(p)


def test_semantic_weights_and_zero_radius_fallback():
    actual=semantic_weights(.25,torch.tensor([3.,0.]),torch.tensor([1.,0.]))
    torch.testing.assert_close(actual,torch.tensor([.5,.25]))
    assert not actual.requires_grad


def test_large_margin_matches_linear_probability_boundary():
    feature=torch.tensor([[.2,.4],[.5,-.3]],requires_grad=True)
    weight=torch.tensor([[1.,0.],[0.,1.]])
    logits=feature@weight.T
    target=torch.tensor([0,1])
    loss=large_margin(logits,target,feature,10)
    prob=logits.softmax(1)
    gap=torch.stack([prob[0,0]-prob[0,1],prob[1,1]-prob[1,0]])
    norm=(2*prob[:,0]*prob[:,1]*np.sqrt(2)).detach()
    distance=gap/(norm+1e-8)
    expected=torch.relu(10-distance.clamp_min(-30))-10
    torch.testing.assert_close(loss,expected)
    loss.mean().backward()
    assert torch.isfinite(feature.grad).all()


def test_contrastive_positive_pairs_and_masks():
    a=torch.eye(4,requires_grad=True)
    assert contrastive(a,a,.1)<contrastive(a,a.flip(0),.1)
    loss=contrastive(a,a,.1,torch.tensor([0,0,1,1]))
    loss.backward()
    assert torch.isfinite(a.grad).all()
    x=torch.randn(4,50,6)
    masked,mask=masked_view(x,np.random.default_rng(13),DEFAULTS['crosshar'])
    assert torch.all(mask.sum(1)==round(50*.15))
    torch.testing.assert_close(masked[~mask],x[~mask])


@pytest.mark.parametrize('method', ['contratsc','sdmix','crosshar'])
def test_method_rejects_silent_overrides(method):
    p=small_pipeline(method)
    p.model['custom_params']['typo_epochs']=1
    with pytest.raises(ValueError,match='settings'):
        validate_pipeline(p,{'uci_har'})
    del p.model['custom_params']['typo_epochs']
    p.augmentation['kind']='jitter'
    with pytest.raises(ValueError,match='own their training'):
        validate_pipeline(p,{'uci_har'})


@pytest.mark.parametrize('method', ['contratsc','sdmix','crosshar'])
def test_comparison_cli_runs_without_llm(method,manifest,tmp_path):
    import subprocess
    import sys
    p=small_pipeline(method)
    p.data['overlap']=.92  
    p.model.update(max_epochs=1,batch_size=128)
    path=tmp_path/f'{method}.json'
    path.write_text(json.dumps(p.payload()))
    out=tmp_path/f'cli-{method}'
    result=subprocess.run([sys.executable,'-m','autohar_researcher.main','search',
        '--manifest',str(manifest),'--protocol','lodo','--target','uci_har',
        '--method',method,'--baseline',str(path),'--seeds','13','--device','cpu',
        '--output',str(out),'--final-test'],capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr
    summary=json.loads((out/'summary.json').read_text())
    assert summary['runs'][0]['evaluations']==1
    assert 0<=summary['test_mean']<=1
    folder=out/'seed-13/uci_har'
    assert (folder/'comparison-final_refit.json').is_file()
    assert 'test_macro_f1' not in (folder/'history.jsonl').read_text()


def test_radii_depend_on_domain_and_class():
    from research.comparisons.training import source_radii
    class Identity(torch.nn.Module):
        def features(self,x):
            return x.flatten(1)
    x=np.array([0.,2.,10.,16.],dtype=np.float32).reshape(4,1,1)
    y=np.zeros(4,dtype=int)
    result=source_radii(Identity(),x,y,np.array(['a','a','b','b']),DEFAULTS['sdmix'],'cpu',2)
    assert float(result[('a',0)])==1
    assert float(result[('b',0)])==3


def test_frozen_representation_does_not_update_batch_norm():
    from research.comparisons.models import CATCC
    from research.comparisons.training import set_trainable
    model=CATCC(6,5,50,DEFAULTS['contratsc'])
    set_trainable(model,False)
    set_trainable(model.head,True)
    model.eval()
    model.head.train()
    before={key:value.clone() for key,value in model.encoder.state_dict().items()}
    opt=torch.optim.Adam(model.head.parameters())
    loss=torch.nn.functional.cross_entropy(model(torch.randn(4,50,6)),torch.tensor([0,1,2,3]))
    loss.backward();opt.step()
    for key,value in model.encoder.state_dict().items():
        torch.testing.assert_close(value,before[key])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('method',['contratsc','sdmix','crosshar'])
def test_comparison_training_on_cuda(method,manifest,tmp_path):
    torch.set_num_threads(1)
    task=make_task(manifest,'lodo','uci_har')
    h=HARHarness(task,13,tmp_path/method,device='cuda')
    p=small_pipeline(method)
    x=np.random.default_rng(3).normal(size=(20,50,6)).astype(np.float32)
    y=np.tile(np.arange(5),4)
    h._comparison_domains=np.repeat(['motion_sense','real_life_har'],10)
    model,epochs=h._fit(p,x,y,epochs=1)
    assert epochs==1
    assert next(model.parameters()).is_cuda
    assert np.isfinite(h._predict(model,x,8)).all()
