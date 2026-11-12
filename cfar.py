"""
cfar.py
-------
Détection à taux de fausse alarme constant (CFAR - Constant False Alarm Rate).

Principe :
    Un seuil de détection fixe ne fonctionne pas en présence de bruit/clutter
    dont le niveau varie dans le temps ou l'espace : trop bas => trop de
    fausses alarmes ; trop haut => cibles manquées.

    Le CFAR estime le niveau de bruit LOCALEMENT autour de chaque cellule
    testée (Cell Under Test, CUT) à partir de cellules d'entraînement
    voisines (training cells), en excluant des cellules de garde (guard
    cells) pour ne pas inclure l'énergie de la cible elle-même dans
    l'estimation du bruit.

    Seuil(i) = alpha * estimate_bruit(i)

    alpha (facteur de seuil) est calculé pour obtenir une probabilité de
    fausse alarme (Pfa) cible, en supposant un bruit avec une statistique
    connue (ici, exponentielle sur la puissance = cas classique du bruit
    gaussien complexe après détection quadratique).

Deux variantes implémentées :
    - CA-CFAR (Cell Averaging) : moyenne arithmétique des cellules
      d'entraînement. Simple, optimal en bruit homogène.
    - OS-CFAR (Ordered Statistic) : k-ième plus grande valeur triée des
      cellules d'entraînement. Plus robuste en présence de cibles
      multiples/clutter hétérogène (moins sensible au "masking").
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class CFARConfig:
    num_train: int = 16     # nb de cellules d'entraînement (de chaque côté)
    num_guard: int = 4      # nb de cellules de garde (de chaque côté)
    pfa: float = 1e-3       # probabilité de fausse alarme visée
    method: str = "CA"      # "CA" ou "OS"
    os_k: int | None = None  # rang k pour OS-CFAR (par défaut ~ 3/4 des cellules)


def _ca_cfar_alpha(num_train_total: int, pfa: float) -> float:
    """
    Facteur de seuil pour CA-CFAR en bruit gaussien complexe (loi
    exponentielle sur la puissance après détection quadratique) :

        alpha = N * (Pfa^(-1/N) - 1)

    avec N = nombre total de cellules d'entraînement.
    Référence classique : Rohling (1983), Skolnik "Radar Handbook".
    """
    N = num_train_total
    return N * (pfa ** (-1.0 / N) - 1.0)


def _os_cfar_alpha(
    num_train_total: int, k: int, pfa: float, mc_samples: int = 200_000
) -> float:
    """
    Facteur de seuil pour OS-CFAR, calibré par Monte-Carlo.

    Le bruit de puissance normalisé suit une loi exponentielle(1) (cas
    gaussien complexe après détection quadratique). Le CUT (cell under
    test) Y et la statistique d'ordre X_(k) (k-ième plus petite valeur
    parmi les N cellules d'entraînement, triées par ordre croissant) sont
    indépendants sous H0. Comme Y ~ Exp(1), P(Y > y) = exp(-y), donc :

        Pfa(alpha) = E[ P(Y > alpha * X_(k)) ] = E[ exp(-alpha * X_(k)) ]

    l'espérance étant prise sur le tirage aléatoire de X_(k). On estime
    cette espérance par Monte-Carlo (tirage direct de N échantillons
    Exponentiel(1), tri, extraction du k-ième), puis on résout
    Pfa(alpha) = pfa par recherche de racine (bisection).

    Cette approche évite la formule fermée de Rohling (produits de
    fonctions Beta), plus délicate à implémenter correctement, tout en
    restant exacte à l'erreur Monte-Carlo près (négligeable avec
    mc_samples grand).
    """
    from scipy.optimize import brentq

    N = num_train_total
    rng = np.random.default_rng(12345)  # graine fixe : seuil reproductible
    samples = rng.exponential(scale=1.0, size=(mc_samples, N))
    samples.sort(axis=1)
    x_k = samples[:, k - 1]  # k-ième plus petite valeur (1-indexé)

    def pfa_of_alpha(alpha):
        return np.mean(np.exp(-alpha * x_k)) - pfa

    try:
        alpha = brentq(pfa_of_alpha, 1e-6, 1e4)
    except ValueError:
        # fallback si la bisection échoue (cas extrêmes) : repli CA-CFAR
        alpha = _ca_cfar_alpha(N, pfa)
    return alpha


def cfar_detector(
    magnitude: np.ndarray, config: CFARConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applique un détecteur CFAR 1D (CA-CFAR ou OS-CFAR) le long d'un profil
    (typiquement le profil distance en puissance, ou en amplitude^2).

    Parameters
    ----------
    magnitude : ndarray
        Profil d'amplitude (PAS en dB) — ex: |signal comprimé|.
    config : CFARConfig

    Returns
    -------
    detections : ndarray (bool)
        Masque des cellules détectées comme cible.
    threshold : ndarray (float)
        Seuil adaptatif calculé à chaque position (même unité que `magnitude`).
    """
    power = magnitude ** 2  # on travaille en puissance (statistique exponentielle)
    n = len(power)

    ng = config.num_guard
    nt = config.num_train
    win = ng + nt

    threshold = np.full(n, np.inf)
    detections = np.zeros(n, dtype=bool)

    num_train_total = 2 * nt

    if config.method.upper() == "CA":
        alpha = _ca_cfar_alpha(num_train_total, config.pfa)
    elif config.method.upper() == "OS":
        k = config.os_k or int(0.75 * num_train_total)
        alpha = _os_cfar_alpha(num_train_total, k, config.pfa)
        k_idx = k
    else:
        raise ValueError(f"méthode CFAR inconnue: {config.method}")

    for i in range(win, n - win):
        lead_train = power[i - win : i - ng]
        lag_train = power[i + ng + 1 : i + win + 1]
        train_cells = np.concatenate([lead_train, lag_train])

        if config.method.upper() == "CA":
            noise_est = np.mean(train_cells)
        else:  # OS
            sorted_cells = np.sort(train_cells)
            noise_est = sorted_cells[k_idx - 1]

        thresh_val = alpha * noise_est
        threshold[i] = thresh_val
        if power[i] > thresh_val:
            detections[i] = True

    return detections, np.sqrt(threshold)  # on repasse en unité "amplitude"
