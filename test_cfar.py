import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.cfar import CFARConfig, cfar_detector


def test_ca_cfar_detects_single_spike_in_noise():
    rng = np.random.default_rng(0)
    n = 500
    noise_power = 1.0
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )
    magnitude = np.abs(noise)

    spike_idx = 250
    magnitude[spike_idx] += 15.0  # cible très au-dessus du bruit

    cfg = CFARConfig(num_train=16, num_guard=4, pfa=1e-3, method="CA")
    detections, threshold = cfar_detector(magnitude, cfg)

    assert detections[spike_idx]
    # peu de fausses alarmes attendu vu la Pfa faible
    assert detections.sum() < 10


def test_cfar_false_alarm_rate_is_low_on_pure_noise():
    rng = np.random.default_rng(1)
    n = 4000
    noise = np.sqrt(0.5) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    magnitude = np.abs(noise)

    cfg = CFARConfig(num_train=16, num_guard=4, pfa=1e-3, method="CA")
    detections, _ = cfar_detector(magnitude, cfg)

    fa_rate = detections.sum() / n
    # la Pfa visée est 1e-3 ; on tolère une marge (échantillon fini)
    assert fa_rate < 1e-2


def test_os_cfar_runs_and_returns_expected_shapes():
    rng = np.random.default_rng(2)
    n = 300
    magnitude = np.abs(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    cfg = CFARConfig(num_train=16, num_guard=4, pfa=1e-2, method="OS")
    detections, threshold = cfar_detector(magnitude, cfg)

    assert detections.shape == (n,)
    assert threshold.shape == (n,)
