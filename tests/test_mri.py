import numpy as np
import pytest

from cardiac import config
from cardiac.mri import preprocess


def test_parse_info_cfg(tmp_path):
    cfg = tmp_path / "Info.cfg"
    cfg.write_text("ED: 1\nES: 12\nGroup: DCM\nHeight: 184.0\nNbFrame: 30\nWeight: 95.0\n")
    info = preprocess.parse_info_cfg(cfg)
    assert info == {"ED": "1", "ES": "12", "Group": "DCM", "Height": "184.0",
                    "NbFrame": "30", "Weight": "95.0"}


@pytest.mark.parametrize("shape", [(216, 256, 10), (154, 224, 6), (128, 128, 8)])
def test_resize_volume_exact_shape(shape):
    vol = np.random.default_rng(0).random(shape).astype(np.float32)
    assert preprocess.resize_volume(vol).shape == config.MRI_TARGET_SHAPE


def test_zscore():
    v = preprocess.zscore(np.random.default_rng(0).normal(300, 50, (16, 16, 4)))
    assert abs(v.mean()) < 1e-4 and abs(v.std() - 1) < 1e-3


def test_heart_metrics_and_ejection_fraction():
    ed = np.zeros((10, 10, 10), dtype=np.int16)
    es = np.zeros_like(ed)
    ed[:5, :4] = 3
    es[:5, :2] = 3
    ed[5:, :2] = 1
    es[5:, :1] = 1
    m = preprocess.heart_metrics(ed, es, (1.0, 1.0, 10.0), {"Height": "170", "Weight": "70"})
    assert m["EDV_LV"] == pytest.approx(2.0)
    assert m["EF_LV"] == pytest.approx(50.0)
    assert m["EF_RV"] == pytest.approx(50.0)
    assert "EDV_LV_index" in m


def test_class_order_is_single_source_of_truth():
    assert config.ACDC_CLASSES == ["NOR", "MINF", "DCM", "HCM", "RV"]
    assert set(config.ACDC_CLASS_NAMES) == set(config.ACDC_CLASSES)
