import numpy as np


def impedance(frequency, rs, rp, q, n):
    """Rs + (Rp || CPE); frequency in Hz, resistance in ohms."""
    w = 2 * np.pi * np.asarray(frequency, dtype=float)
    return rs + 1 / (1 / rp + q * (1j * w) ** n)
