# Détection radar par filtrage adapté (matched filter) + CFAR

Simulation d'une chaîne de traitement radar classique en Python :
génération d'une impulsion chirp (LFM), compression d'impulsion par
filtrage adapté, et détection de cibles à taux de fausse alarme constant
(CFAR) en présence de bruit et de clutter.

Projet pédagogique , sans dépendance à du matériel radar réel :
tout est simulé numériquement (numpy/scipy), avec des paramètres physiques
réalistes (fréquence d'échantillonnage, bande passante, Nyquist).

## Principe

```
 Chirp émis (LFM)                         Signal reçu (fenêtre d'écoute)
 s_tx(t) = exp(j·π·k·t²)     -->     Σ échos retardés (2R/c) + Doppler + bruit
                                                    |
                                                    v
                                    Filtrage adapté (corrélation, via FFT)
                                                    |
                                                    v
                                    Profil distance (pic étroit par cible)
                                                    |
                                                    v
                                    CFAR : seuil adaptatif local -> détections
```

Le filtre adapté à une impulsion `s_ref` est `h(t) = conj(s_ref(-t))`. Le
filtrer par un signal reçu revient à corréler le signal reçu avec `s_ref` :
c'est le traitement qui maximise le SNR en sortie pour un bruit blanc
gaussien, et qui "comprime" un écho chirp étalé dans le temps en un pic
étroit. La position de ce pic code directement le délai aller-retour,
donc la distance de la cible : `R = k · c / (2·fs)`.

Le CFAR résout un problème classique : un seuil de détection fixe ne
fonctionne pas quand le niveau de bruit/clutter varie. Le CFAR estime le
bruit **localement**, à partir de cellules d'entraînement voisines de la
cellule testée (en excluant des cellules de garde pour ne pas inclure
l'énergie de la cible dans l'estimation), et fixe un seuil proportionnel à
cette estimation pour garantir une Pfa constante :

```
Seuil(i) = alpha · bruit_estime(i)
```

- **CA-CFAR** (Cell Averaging) : `bruit_estime` = moyenne des cellules
  d'entraînement. Optimal en bruit homogène, mais sensible au "masking"
  (une cible interférente dans la fenêtre d'entraînement fait monter le
  seuil et peut masquer une cible voisine).
- **OS-CFAR** (Ordered Statistic) : `bruit_estime` = k-ième valeur triée
  des cellules d'entraînement. Plus robuste en clutter hétérogène /
  cibles multiples, au prix d'un coût de calcul un peu plus élevé.

Le facteur `alpha` est calculé pour respecter une Pfa cible, en supposant
un bruit gaussien complexe (donc une puissance qui suit une loi
exponentielle) :
- CA-CFAR : formule fermée classique (Rohling 1983) `alpha = N·(Pfa^(-1/N) - 1)`.
- OS-CFAR : calibré par Monte-Carlo (`Pfa(alpha) = E[exp(-alpha·X_(k))]`,
  identité exacte tirée de la loi exponentielle du CUT), plus robuste
  numériquement qu'une formule fermée en produits de fonctions Beta.

## Structure du projet

```
radar-matched-filter-cfar/
├── src/
│   ├── chirp.py           # génération du chirp LFM + simulation de la scène reçue
│   ├── matched_filter.py  # compression d'impulsion (corrélation directe / FFT)
│   └── cfar.py            # détecteurs CA-CFAR et OS-CFAR
├── examples/
│   └── run_demo.py        # démo complète : simulation -> compression -> CFAR -> figures
├── tests/
│   ├── test_chirp.py
│   └── test_cfar.py
├── figures/                # générées par run_demo.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/saif1eddine1ghnimi/radar-matched-filter-cfar.git
cd radar-matched-filter-cfar
pip install -r requirements.txt
```

## Utilisation

```bash
python examples/run_demo.py
```

Génère 4 figures dans `figures/` :
1. `01_chirp_waveform.png` — chirp émis (I/Q, fréquence instantanée).
2. `02_range_profile.png` — profil distance après compression d'impulsion.
3. `03_ca_cfar_detection.png` — profil + seuil CA-CFAR + détections.
4. `04_ca_vs_os_cfar_zoom.png` — zoom sur deux cibles proches, CA-CFAR vs OS-CFAR.


## Tests

```bash
pip install pytest
pytest tests/ -v
```

Les tests vérifient notamment :
- que la compression d'impulsion retrouve bien la distance d'une cible
  isolée (à la résolution théorique près) ;
- que les implémentations "directe" et "FFT" du filtrage adapté
  concordent numériquement ;
- que le taux de fausse alarme mesuré du CFAR sur du bruit pur est
  conforme à la Pfa visée.

## Limites connues 

- Le délai de chaque cible est arrondi à l'échantillon le plus proche
  (pas d'interpolation sous-échantillon) — acceptable tant que `1/fs`
  reste petit devant la résolution radar `c/(2B)`.
- Pas de traitement Doppler multi-impulsions (pas de FFT lente / carte
  distance-vitesse) : le Doppler n'est ici qu'un déphasage appliqué à
  l'écho d'une impulsion unique, illustratif mais non exploité pour la
  détection.
- Le clutter est modélisé comme des points diffuseurs supplémentaires
  (pas de modèle statistique de clutter distribué, ex. Weibull/K-distribution).

## Références

- Rohling, H. (1983). *Radar CFAR Thresholding in Clutter and Multiple
  Target Situations*. IEEE Transactions on Aerospace and Electronic Systems.
- Skolnik, M. (2008). *Radar Handbook*, 3rd edition.
- Richards, M. A. (2014). *Fundamentals of Radar Signal Processing*.
