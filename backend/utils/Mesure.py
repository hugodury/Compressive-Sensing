"""
Acquisition compressée et métriques associées (spec fonction.md).

Les implémentations vivent dans `mesure.py` à la racine du dépôt
pour éviter la duplication ; ce module ré-exporte l’API attendue.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mesure import (  # noqa: E402
    apply_measurement,
    compute_coherence,
    compute_ratio,
    encoder_bcs,
    generate_measurement_matrix,
)

__all__ = [
    "apply_measurement",
    "compute_coherence",
    "compute_ratio",
    "encoder_bcs",
    "generate_measurement_matrix",
]
