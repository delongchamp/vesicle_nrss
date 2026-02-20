"""Result dataclass for vesicle runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VesicleResults:
    data: Any
    remeshed_data: Any
    I: Any
    I_para: Any
    I_perp: Any
    A: Any
    model_plots: Any
    args: Any
    accepted_radii_nm: np.ndarray
