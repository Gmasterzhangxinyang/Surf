from __future__ import annotations

import numpy as np


def estimate_local_ground_z(points_xyz: np.ndarray, quantile: float = 0.08) -> float:
    if len(points_xyz) == 0:
        return 0.0
    z = np.asarray(points_xyz[:, 2], dtype=np.float64)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return 0.0
    return float(np.quantile(z, quantile))
