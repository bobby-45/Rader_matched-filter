"""
run_demo.py
-----------
Démo complète de la chaîne de traitement radar :

    1. Génération d'une impulsion chirp (LFM).
    2. Simulation du signal reçu sur la fenêtre d'écoute : cibles utiles
       + clutter dense + bruit.
    3. Compression d'impulsion par filtrage adapté (matched filter, via FFT).
    4. Détection CFAR (CA-CFAR et OS-CFAR) sur le profil distance.
    5. Génération de figures (sauvegardées dans ../figures/).

Lancer depuis la racine du projet :
    python examples/run_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.chirp import RadarParams, Target, generate_chirp, simulate_received_signal
from src.matched_filter import matched_filter, range_profile
from src.cfar import CFARConfig, cfar_detector

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    params = RadarParams(fs=200e6, T_chirp=1e-6, B=100e6, fc=24e9, R_max=300.0)

    print(f"Résolution distance théorique : {params.range_resolution:.2f} m")
    print(f"Portée max observable          : {params.max_range:.1f} m")
    print(f"Échantillons par impulsion     : {params.n_chirp}")
    print(f"Échantillons fenêtre d'écoute  : {params.n_obs}")

    # --- 1. Chirp émis (référence) ---
    t_tx, s_tx = generate_chirp(params)

    # --- 2. Scène radar : cibles utiles + clutter dense ---
    targets = [
        Target(range_m=120.0, velocity_mps=15.0, rcs=1.0),
        Target(range_m=180.0, velocity_mps=-5.0, rcs=0.6),
        Target(range_m=186.0, velocity_mps=2.0, rcs=0.5),  # cible proche -> test résolution
    ]
    rng = np.random.default_rng(42)
    clutter = [
        Target(range_m=r, velocity_mps=0.0, rcs=0.15)
        for r in rng.uniform(20, 280, size=25)
    ]

    t_rx, s_rx = simulate_received_signal(
        params, targets, snr_db=13.0, clutter_targets=clutter, seed=1
    )

    # --- 3. Compression d'impulsion (filtrage adapté) ---
    compressed = matched_filter(s_rx, s_tx, mode="fft")
    ranges, mag_db = range_profile(params, compressed)
    mag_linear = 10 ** (mag_db / 20)

    # --- 4. Détection CFAR ---
    cfg_ca = CFARConfig(num_train=20, num_guard=6, pfa=1e-4, method="CA")
    det_ca, thresh_ca = cfar_detector(mag_linear, cfg_ca)

    cfg_os = CFARConfig(num_train=20, num_guard=6, pfa=1e-4, method="OS")
    det_os, thresh_os = cfar_detector(mag_linear, cfg_os)

    thresh_ca_db = 20 * np.log10(thresh_ca + 1e-12)
    thresh_os_db = 20 * np.log10(thresh_os + 1e-12)

    print("\nCibles réelles (m)     :", [tgt.range_m for tgt in targets])
    print("Détections CA-CFAR (m) :", np.round(ranges[det_ca], 2).tolist())
    print("Détections OS-CFAR (m) :", np.round(ranges[det_os], 2).tolist())

    # --- 5. Figures ---
    plt.style.use("default")

    # Figure 1 : chirp émis (temps + fréquence instantanée)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    axes[0].plot(t_tx * 1e6, np.real(s_tx), label="I (real)", lw=1.0)
    axes[0].plot(t_tx * 1e6, np.imag(s_tx), label="Q (imag)", lw=1.0, alpha=0.7)
    axes[0].set_title("Impulsion chirp (LFM) émise, bande de base")
    axes[0].set_xlabel("Temps (µs)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    f_inst = params.k * t_tx
    axes[1].plot(t_tx * 1e6, f_inst / 1e6, color="darkorange")
    axes[1].set_title("Fréquence instantanée du chirp")
    axes[1].set_xlabel("Temps (µs)")
    axes[1].set_ylabel("Fréquence (MHz)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_chirp_waveform.png"), dpi=150)

    # Figure 2 : profil distance brut vs comprimé
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ranges, mag_db, lw=1.0, label="Profil comprimé")
    for tgt in targets:
        ax.axvline(tgt.range_m, color="green", ls="--", lw=1, alpha=0.6)
    ax.set_title("Profil distance après compression d'impulsion (filtrage adapté)")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_xlim(0, params.max_range)
    ax.legend(["Profil comprimé", "Cibles réelles"])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_range_profile.png"), dpi=150)

    # Figure 3 : CFAR (CA) — profil + seuil adaptatif + détections
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ranges, mag_db, lw=1.0, label="Profil distance", color="steelblue")
    ax.plot(ranges, thresh_ca_db, lw=1.2, label="Seuil CA-CFAR", color="crimson")
    ax.scatter(
        ranges[det_ca], mag_db[det_ca], color="black", marker="x", s=60,
        label="Détections CA-CFAR", zorder=5,
    )
    for tgt in targets:
        ax.axvline(tgt.range_m, color="green", ls="--", lw=1, alpha=0.5)
    ax.set_title(f"Détection CA-CFAR (Pfa={cfg_ca.pfa:.0e})")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_xlim(0, params.max_range)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_ca_cfar_detection.png"), dpi=150)

    # Figure 4 : comparaison CA-CFAR vs OS-CFAR (zoom cibles proches)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ranges, mag_db, lw=1.0, label="Profil distance", color="steelblue")
    ax.plot(ranges, thresh_ca_db, lw=1.2, label="Seuil CA-CFAR", color="crimson")
    ax.plot(ranges, thresh_os_db, lw=1.2, label="Seuil OS-CFAR", color="darkorange")
    ax.scatter(ranges[det_ca], mag_db[det_ca], color="black", marker="x", s=60,
               label="Détections CA-CFAR", zorder=5)
    ax.scatter(ranges[det_os], mag_db[det_os], color="purple", marker="+", s=80,
               label="Détections OS-CFAR", zorder=5)
    ax.set_title("Zoom : cibles proches (170-195 m) — CA-CFAR vs OS-CFAR")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_xlim(170, 195)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_ca_vs_os_cfar_zoom.png"), dpi=150)

    print(f"\nFigures sauvegardées dans : {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    main()
