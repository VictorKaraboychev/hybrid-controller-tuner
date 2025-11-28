"""
Utility functions for polynomial operations and other common tasks.
"""

import numpy as np
from numpy.polynomial import polynomial


def roots_to_coefficients(roots: list[float]) -> list[float]:
    """
    Convert polynomial roots to coefficients in descending order.

    Given roots [p0, p1, ..., pn], returns coefficients [cn, cn-1, ..., c1, c0]
    such that:
        (x - p0)(x - p1)...(x - pn) = c_n * x^n + c_{n-1} * x^{n-1} + ... + c_1 * x + c_0

    Parameters
    ----------
    roots : list[float]
        List of polynomial roots [p0, p1, ..., pn]

    Returns
    -------
    list[float]
        Coefficients in descending order [cn, cn-1, ..., c1, c0] where:
        - cn is the coefficient of x^n (leading coefficient)
        - cn-1 is the coefficient of x^(n-1)
        - c0 is the constant term

    Examples
    --------
    >>> roots_to_coefficients([1, 2])
    [1.0, -3.0, 2.0]
    # Represents: (x - 1)(x - 2) = x^2 - 3x + 2 = 1*x^2 + (-3)*x + 2

    >>> roots_to_coefficients([-1, 0, 1])
    [1.0, 0.0, -1.0, 0.0]
    # Represents: (x + 1)(x - 0)(x - 1) = x^3 - x = 1*x^3 + 0*x^2 + (-1)*x + 0
    """
    if not roots:
        return [1.0]  # Empty product is 1

    # Convert to numpy array and compute polynomial coefficients
    roots_array = np.array(roots, dtype=float)
    coefficients = polynomial.polyfromroots(roots_array)

    # Reverse to get descending order (highest degree first)
    return coefficients.tolist()[::-1]

