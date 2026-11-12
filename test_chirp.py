import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.chirp import RadarParams, Target, generate_chirp, simulate_received_signal
from src.matched_filter import matched_filter, range_profile


def test_chirp_length_and_unit_modulus():
    params = RadarParams(fs=200e6, T_chirp=1e-6, B=100e6)
    t, s_tx = generate_chirp(params)
    assert len(t) == params.n_chirp
    # Le chirp est à module constant (== 1) par construction.
    assert np.allclose(np.abs(s_tx), 1.0, atol=1e-9)


def test_matched_filter_detects_single_target_range():
    params = RadarParams(fs=200e6, T_chirp=1e-6, B=100e6, R_max=300.0)
    _, s_tx = generate_chirp(params)

    true_range = 150.0
    targets = [Target(range_m=true_range, velocity_mps=0.0, rcs=1.0)]
    _, s_rx = simulate_received_signal(params, targets, snr_db=30.0, seed=0)

    compressed = matched_filter(s_rx, s_tx, mode="fft")
    ranges, mag_db = range_profile(params, compressed)

    peak_range = ranges[np.argmax(mag_db)]
    # tolérance = 2x résolution théorique (arrondi à l'échantillon près)
    assert abs(peak_range - true_range) < 2 * params.range_resolution


def test_fft_and_direct_matched_filter_agree():
    params = RadarParams(fs=200e6, T_chirp=1e-6, B=100e6, R_max=100.0)
    _, s_tx = generate_chirp(params)
    targets = [Target(range_m=60.0, velocity_mps=0.0, rcs=1.0)]
    _, s_rx = simulate_received_signal(params, targets, snr_db=30.0, seed=0)

    out_fft = matched_filter(s_rx, s_tx, mode="fft")
    out_direct = matched_filter(s_rx, s_tx, mode="direct")

    assert len(out_fft) == len(out_direct) == len(s_rx)
    assert np.argmax(np.abs(out_fft)) == np.argmax(np.abs(out_direct))
    assert np.allclose(out_fft, out_direct, atol=1e-6)


def test_nyquist_violation_raises():
    try:
        RadarParams(fs=50e6, B=100e6)  # fs < B => doit lever une erreur
        raised = False
    except ValueError:
        raised = True
    assert raised
