"""
chirp.py
--------
Génération de la forme d'onde radar à modulation linéaire de fréquence
(LFM chirp, la forme d'onde utilisée aussi bien en FMCW qu'en radar à
impulsions comprimées) et simulation de la scène radar reçue (cibles +
clutter + bruit) sur une fenêtre d'écoute.

Modèle utilisé : radar à impulsion chirpée (pulse-compression radar).
    - On émet UNE impulsion chirp de durée T_chirp.
    - On écoute pendant une fenêtre de durée T_obs > T_chirp, pendant
      laquelle reviennent les échos retardés des cibles (delay = 2R/c).
    - Chaque écho est une copie retardée, atténuée et déphasée (Doppler)
      de l'impulsion émise, noyée dans du bruit.
    - La compression d'impulsion (matched_filter.py) corrèle le signal
      reçu avec l'impulsion de référence : chaque écho redevient un pic
      étroit dont la position temporelle code directement la distance.

C'est le même principe physique qu'un radar FMCW pulsé à rampe courte,
et c'est le cas d'usage canonique du filtrage adapté en traitement radar.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class RadarParams:
    """Paramètres du radar (forme d'onde chirp + fenêtre d'observation)."""

    fs: float = 200e6         # fréquence d'échantillonnage (Hz) -- doit respecter Nyquist (fs > B)
    T_chirp: float = 1e-6     # durée de l'impulsion chirp (s)
    B: float = 100e6          # bande passante balayée par le chirp (Hz)
    fc: float = 24e9          # fréquence porteuse (Hz), utile pour le calcul Doppler
    c: float = 3e8            # vitesse de la lumière (m/s)
    R_max: float = 300.0      # portée max que l'on souhaite observer (m)

    def __post_init__(self):
        if self.fs <= self.B:
            raise ValueError(
                "fs doit être > B pour respecter Nyquist et éviter le repliement "
                f"du chirp (fs={self.fs:.2e}, B={self.B:.2e})."
            )

    @property
    def k(self) -> float:
        """Pente du chirp (Hz/s)."""
        return self.B / self.T_chirp

    @property
    def n_chirp(self) -> int:
        """Nombre d'échantillons de l'impulsion émise."""
        return int(round(self.T_chirp * self.fs))

    @property
    def n_delay_max(self) -> int:
        """Nombre d'échantillons correspondant au délai aller-retour max (R_max)."""
        tau_max = 2 * self.R_max / self.c
        return int(round(tau_max * self.fs))

    @property
    def n_obs(self) -> int:
        """Longueur totale de la fenêtre d'écoute (échantillons)."""
        return self.n_delay_max + self.n_chirp

    @property
    def range_resolution(self) -> float:
        """Résolution en distance théorique = c / (2*B)."""
        return self.c / (2 * self.B)

    @property
    def max_range(self) -> float:
        """Portée max réellement observable compte tenu de l'arrondi n_delay_max."""
        return self.n_delay_max / self.fs * self.c / 2


def generate_chirp(params: RadarParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère l'impulsion chirp LFM émise, en bande de base.

    Returns
    -------
    t : ndarray
        Vecteur temps de l'impulsion (s), longueur n_chirp.
    s_tx : ndarray (complex)
        Signal émis complexe (I/Q), s_tx(t) = exp(j*pi*k*t^2), module 1.
    """
    n = params.n_chirp
    t = np.arange(n) / params.fs
    s_tx = np.exp(1j * np.pi * params.k * t ** 2)
    return t, s_tx


@dataclass
class Target:
    """Cible radar ponctuelle."""

    range_m: float              # distance (m)
    velocity_mps: float = 0.0   # vitesse radiale (m/s), positif = s'éloigne
    rcs: float = 1.0            # amplitude relative (proxy de la SER)


def simulate_received_signal(
    params: RadarParams,
    targets: list[Target],
    snr_db: float | None = 15.0,
    clutter_targets: list[Target] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simule le signal reçu sur la fenêtre d'écoute complète : superposition
    des échos retardés de chaque cible (utile + clutter), plus bruit blanc
    gaussien complexe.

    Pour une cible à distance R et vitesse radiale v :
        - délai aller-retour : tau = 2R/c
        - décalage Doppler   : f_d = 2*v*fc/c
        - l'écho reçu est une copie de l'impulsion émise, retardée de tau,
          multipliée par exp(j*2*pi*f_d*t) et par l'amplitude `rcs`.

    Limite volontairement simplifiée : le délai est arrondi à l'échantillon
    le plus proche (pas d'interpolation sous-échantillon), ce qui est
    suffisant tant que la résolution spatiale d'un échantillon
    (c / (2*fs)) reste petite devant la résolution du radar (c / (2*B)).

    Parameters
    ----------
    params : RadarParams
    targets : list[Target]
        Cibles "utiles" à détecter.
    snr_db : float or None
        SNR (dB) d'une cible de référence (rcs=1) par rapport au bruit.
        None => pas de bruit ajouté.
    clutter_targets : list[Target] or None
        Cibles de fouillis (clutter), typiquement nombreuses et de faible
        rcs, simulées de la même façon que des cibles.
    seed : int
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    t : ndarray
        Vecteur temps de la fenêtre d'écoute (s), longueur n_obs.
    received : ndarray (complex)
        Signal reçu bruité, longueur n_obs.
    """
    rng = np.random.default_rng(seed)
    n_obs = params.n_obs
    n_chirp = params.n_chirp
    t_chirp = np.arange(n_chirp) / params.fs
    _, s_tx = generate_chirp(params)

    received = np.zeros(n_obs, dtype=complex)

    all_targets = list(targets) + list(clutter_targets or [])
    for tgt in all_targets:
        tau = 2 * tgt.range_m / params.c
        delay_idx = int(round(tau * params.fs))
        if delay_idx < 0 or delay_idx + n_chirp > n_obs:
            continue  # cible hors de la fenêtre d'observation (au-delà de R_max)

        f_doppler = 2 * tgt.velocity_mps * params.fc / params.c
        doppler_phase = np.exp(1j * 2 * np.pi * f_doppler * t_chirp)
        received[delay_idx : delay_idx + n_chirp] += tgt.rcs * s_tx * doppler_phase

    if snr_db is not None:
        ref_power = np.mean(np.abs(s_tx) ** 2)  # puissance d'une cible rcs=1
        noise_power = ref_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(n_obs) + 1j * rng.standard_normal(n_obs)
        )
        received = received + noise

    t = np.arange(n_obs) / params.fs
    return t, received
