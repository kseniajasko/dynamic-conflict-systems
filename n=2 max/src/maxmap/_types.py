"""Type aliases shared by the package."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
"""Array of double-precision floats."""

BoolArray: TypeAlias = npt.NDArray[np.bool_]
"""Boolean mask."""

IntArray: TypeAlias = npt.NDArray[np.int64]
"""Array of 64-bit integers."""

FloatOrArray: TypeAlias = float | FloatArray
"""Either a scalar or an array of doubles; used for parameters that broadcast."""
