"""
matched_filter.py
------------------
Compression d'impulsion par filtrage adapté (matched filter).

Le filtre adapté à un signal de référence s_ref est h(t) = conj(s_ref(-t)).
Filtrer un signal reçu par h revient à corréler le signal reçu avec s_ref :
c'est la technique qui maximise le SNR en sortie pour un bruit blanc, et
qui "comprime" un écho chirp étalé dans le temps en un pic étroit dont la
position code le délai (donc la distance) de la cible.

On l'implémente de deux façons équivalentes :
    - "direct" : corrélation directe (np.correlate), sert de référence
      de validation.
    - "fft"    : corrélation via la FFT (théorème de convolution), en
      O(N log N) au lieu de O(N*M) -- c'est la méthode utilisée en
      pratique pour des signaux longs.

Convention d'indexation (identique pour les deux modes) :
    compressed[k] = sum_n received[n+k] * conj(reference[n])
Le pic à l'indice k correspond à un délai de k échantillons entre le
signal reçu et la référence, donc à une distance R = k * c / (2*fs).
"""

from __future__ import annotations

import numpy as np


def matched_filter(
    received: np.ndarray, reference: np.ndarray, mode: str = "fft"
) -> np.ndarray:
    """
    Applique un filtrage adapté (corrélation) entre le signal reçu et le
    signal de référence (l'impulsion chirp émise).

    Parameters
    ----------
    received : ndarray (complex)
        Signal reçu (bruité), longueur N (fenêtre d'écoute complète).
    reference : ndarray (complex)
        Signal de référence (chirp émis), longueur M <= N.
    mode : {"fft", "direct"}
        "fft"    : corrélation via FFT (recommandé, O(N log N)).
        "direct" : corrélation directe np.correlate (référence de test).

    Returns
    -------
    compressed : ndarray (complex)
        Signal comprimé, longueur N. compressed[k] correspond à un délai
        de k échantillons (voir convention ci-dessus).
    """
    n = len(received)
    m = len(reference)

    if mode == "direct":
        # np.correlate(a, v, 'full')[j] = sum_n a[n + j - (M-1)] * conj(v[n])
        # => pour retrouver compressed[k] = sum_n received[n+k]*conj(ref[n]),
        #    on prend j = k + (M-1), donc on découpe à partir de start=M-1.
        corr = np.correlate(received, reference, mode="full")
        start = m - 1
        return corr[start : start + n]

    elif mode == "fft":
        nfft = n + m - 1  # zero-padding => corrélation LINÉAIRE (pas de repliement)
        R = np.fft.fft(received, n=nfft)
        S = np.fft.fft(reference, n=nfft)
        corr_full = np.fft.ifft(R * np.conj(S))
        # Avec ce zero-padding, compressed[k] pour k=0..n-1 se lit
        # directement aux indices 0..n-1 du résultat (pas de décalage).
        return corr_full[:n]

    else:
        raise ValueError(f"mode inconnu: {mode!r}")


def range_axis(params, n_samples: int | None = None) -> np.ndarray:
    """
    Axe des distances (m) associé aux échantillons du signal comprimé.

    L'indice k du signal comprimé correspond à un délai tau = k / fs,
    donc à une distance R = tau * c / 2 = k * c / (2*fs).
    """
    n = n_samples or params.n_obs
    idx = np.arange(n)
    return idx * params.c / (2 * params.fs)


def range_profile(params, compressed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convertit la sortie du filtre adapté en profil distance (range profile),
    restreint à la portée max observable [0, params.max_range].

    Returns
    -------
    ranges : ndarray
        Distances (m), croissantes.
    magnitude_db : ndarray
        Amplitude du profil comprimé en dB.
    """
    n = len(compressed)
    ranges = range_axis(params, n)
    mag = np.abs(compressed)

    valid = ranges <= params.max_range
    ranges_valid = ranges[valid]
    mag_valid = mag[valid]

    mag_db = 20 * np.log10(mag_valid + 1e-12)
    return ranges_valid, mag_db
