import numpy as np
import pytest

from cardiac import config

torch = pytest.importorskip("torch")


def test_cnn_forward_and_backward():
    from cardiac.mri.model import CardiacCNN3D

    model = CardiacCNN3D(num_classes=5, in_channels=2, widths=(4, 8, 8, 16))
    x = torch.randn(2, 2, *(config.MRI_TARGET_SHAPE[2], 32, 32))
    out = model(x)
    assert out.shape == (2, 5)
    out.sum().backward()


def test_dataset_drops_unlabelled_and_augments():
    from cardiac.mri.dataset import ACDCVolumes

    X = np.random.default_rng(0).random((4, 2, 8, 16, 16)).astype(np.float32)
    y = np.array([0, 1, -1, 4])
    ds = ACDCVolumes(X, y, augment=True)
    assert len(ds) == 3
    v, label = ds[0]
    assert tuple(v.shape) == (2, 8, 16, 16) and label.dtype == torch.int64
