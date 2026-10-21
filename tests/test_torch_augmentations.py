import numpy as np
import pytest
import torch
from research.augmentations import CATALOG, MIXING
from research.torch_augmentations import augment, mix_batch, spline, dtw_path
from autohar_researcher.torch_data import batches

DEVICES=['cpu']+(['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('kind',sorted(set(CATALOG)-MIXING))
def test_device_seed_shape_and_no_mutation(device,kind):
    x=torch.randn(4,24,6,device=device)
    original=x.clone()
    y=torch.tensor([0,0,1,1],device=device)
    def run(probability=1):
        return augment(x,{'kind':kind,'probability':probability},
                       torch.Generator(device=device).manual_seed(13),y)
    a,b=run(),run()
    assert a.device==x.device and a.shape==x.shape and torch.isfinite(a).all()
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(x,original)
    torch.testing.assert_close(run(0),x)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('kind',sorted(MIXING))
def test_device_mixed_label_contribution(device,kind):
    x=torch.eye(4,device=device)[:,None,:].expand(4,32,4).clone()
    y=torch.arange(4,device=device)
    a,target=mix_batch(x,y,{'kind':kind},torch.Generator(device=device).manual_seed(13),4)
    assert a.device==x.device and target.device==x.device
    torch.testing.assert_close(a.mean(1),target)
    torch.testing.assert_close(target.sum(1),torch.ones(4,device=device))


@pytest.mark.parametrize('knots',[3,4,6])
def test_cubic_spline_matches_scipy(knots):
    from scipy.interpolate import CubicSpline
    values=torch.randn(knots,6)
    expected=CubicSpline(np.linspace(0,31,knots),values.numpy())(np.arange(32))
    np.testing.assert_allclose(spline(values,32).numpy(),expected,atol=2e-6,rtol=2e-5)


def test_dtw_matches_reference_path():
    def reference(a, b, ratio):
        n, m = len(a), len(b)
        costs = np.full((n+1, m+1), np.inf)
        costs[0, 0] = 0
        band = max(1, int(np.ceil(max(n, m)*ratio)))
        for i in range(1, n+1):
            for j in range(max(1, i-band), min(m, i+band)+1):
                costs[i, j] = np.linalg.norm(a[i-1]-b[j-1]) + min(
                    costs[i-1, j-1], costs[i-1, j], costs[i, j-1])
        i, j, path = n, m, []
        while i and j:
            path.append((i-1, j-1))
            direction = np.argmin([costs[i-1, j-1], costs[i-1, j], costs[i, j-1]])
            i -= direction != 2
            j -= direction != 1
        return np.array(path[::-1]).T
    x=torch.randn(24,3)
    y=torch.randn(24,3)
    np.testing.assert_array_equal(dtw_path(x,y,.1),reference(x.numpy(),y.numpy(),.1))


@pytest.mark.parametrize('device',DEVICES)
def test_pytorch_loader_preserves_alignment_and_moves_batches(device):
    x=np.broadcast_to(np.arange(9)[:,None,None],(9,12,3)).copy()
    seen=[]
    for batch,labels in batches(x,np.arange(9),batch_size=4,device=device,shuffle=True):
        assert batch.device.type==device
        torch.testing.assert_close(batch[:,0,0],labels.float())
        seen.extend(labels.cpu().tolist())
    assert sorted(seen)==list(range(9))
