import numpy as np


def interpolate_data(values: np.ndarray) -> np.ndarray:
    """
    Interpolates data to fill NaN values.

    Args:
        values: The data to be interpolated.

    Returns:
        The interpolated data.
    """
    nans = np.isnan(values)
    idx = lambda z: np.nonzero(z)[0]
    out = np.copy(values)
    out[nans] = np.interp(idx(nans), idx(~nans), values[~nans]) if not np.isnan(values).all() else 0

    return out