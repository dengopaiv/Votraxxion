"""Votrax SC-01A analog filter construction using bilinear z-transform.

All filter designs and capacitor values from MAME votrax.cpp die analysis.
"""

import math
import numpy as np

# Clock constants
MASTER_CLOCK = 720_000
SCLOCK = MASTER_CLOCK // 18   # 40000 Hz - analog sample rate
CCLOCK = SCLOCK // 2          # 20000 Hz - chip update rate

# Combined R*C scaling factor.
# Cap values from MAME die analysis are in abstract units.
# This factor converts cap_value -> RC time constant (seconds).
# Chosen so that formant filter center frequencies fall in the speech range
# (F1: 200-900 Hz, F2: 500-2500 Hz, F3: 1500-3500 Hz, F4: ~2000 Hz).
_RC_SCALE = 1e-8


def bits_to_caps(value: int, caps: tuple) -> float:
    """Convert a multi-bit value to a summed capacitance.

    Each bit in `value` selects the corresponding capacitor from `caps`.
    Bit 0 selects caps[0], bit 1 selects caps[1], etc.
    """
    total = 0.0
    for i, cap in enumerate(caps):
        if value & (1 << i):
            total += cap
    return total


def build_standard_filter(c1t: float, c1b: float, c2t: float, c2b: float,
                          c3: float, c4: float):
    """Build a 2nd-order bandpass/resonator filter (F1, F2v, F3, F4).

    Returns (a, b) as numpy arrays of length 4, implementing a biquad via
    bilinear transform with pre-warping. Based on MAME's active-RC topology.
    """
    if c3 == 0 or c4 == 0:
        return np.zeros(4), np.array([1.0, 0.0, 0.0, 0.0])

    # Integrator angular frequencies
    w3 = 1.0 / (_RC_SCALE * c3)
    w4 = 1.0 / (_RC_SCALE * c4)

    # Center frequency (Hz)
    wn = math.sqrt(w3 * w4)
    fn = wn / (2.0 * math.pi)

    # Pre-warp for bilinear transform
    cy = math.tan(math.pi * fn / SCLOCK)
    cy2 = cy * cy

    # Damping factor from c2 feedback network
    # cx = 1/(2*Q), where Q is determined by c2t, c2b relative to c3
    if (c2t + c2b) > 0 and c3 > 0:
        cx = c2t * c2b / (2.0 * c3 * (c2t + c2b))
    else:
        cx = 0.5
    cx = max(cx, 0.01)  # Ensure stability

    # Input gain from capacitive divider
    if c1t + c1b > 0:
        gain = c1b / (c1t + c1b)
    elif c1b > 0:
        gain = 1.0
    else:
        gain = 0.0

    # Bilinear transform of 2nd-order bandpass:
    # H(s) = gain * wn * s / (s^2 + 2*cx*wn*s + wn^2)
    denom = 1.0 + 2.0 * cx * cy + cy2

    a = np.zeros(4)
    b = np.zeros(4)

    a[0] = gain * cy / denom
    a[1] = 0.0
    a[2] = -gain * cy / denom
    a[3] = 0.0

    b[0] = 1.0
    b[1] = 2.0 * (cy2 - 1.0) / denom
    b[2] = (1.0 - 2.0 * cx * cy + cy2) / denom
    b[3] = 0.0

    return a, b


def build_noise_shaper_filter(c1: float, c2t: float, c2b: float,
                              c3: float, c4: float):
    """Build the noise shaper filter (2nd-order bandpass).

    Returns (a, b) numpy arrays of length 3.
    """
    if c1 == 0 or c3 == 0 or c4 == 0:
        return np.zeros(3), np.array([1.0, 0.0, 0.0])

    w3 = 1.0 / (_RC_SCALE * c3)
    w4 = 1.0 / (_RC_SCALE * c4)

    wn = math.sqrt(w3 * w4)
    fn = wn / (2.0 * math.pi)

    cy = math.tan(math.pi * fn / SCLOCK)
    cy2 = cy * cy

    if (c2t + c2b) > 0 and c3 > 0:
        cx = c2t * c2b / (2.0 * c3 * (c2t + c2b))
    else:
        cx = 0.5
    cx = max(cx, 0.01)

    gain = c3 / c1 if c1 > 0 else 1.0

    denom = 1.0 + 2.0 * cx * cy + cy2

    a = np.zeros(3)
    b = np.zeros(3)

    a[0] = gain * cy / denom
    a[1] = 0.0
    a[2] = -gain * cy / denom

    b[0] = 1.0
    b[1] = 2.0 * (cy2 - 1.0) / denom
    b[2] = (1.0 - 2.0 * cx * cy + cy2) / denom

    return a, b


def build_lowpass_filter(c1t: float, c1b: float):
    """Build the final output lowpass filter (FX).

    Returns (a, b) numpy arrays of length 2.
    """
    if c1t + c1b == 0:
        return np.zeros(2), np.array([1.0, 0.0])

    # Cutoff frequency
    w = 1.0 / (_RC_SCALE * (c1t + c1b))
    freq = w / (2.0 * math.pi)

    cy = math.tan(math.pi * freq / SCLOCK)
    denom = 1.0 + cy

    # Bilinear 1st-order lowpass: H(z) = K*(1+z^-1)/(1+p*z^-1)
    a = np.array([cy / denom, cy / denom])
    b = np.array([1.0, (cy - 1.0) / denom])

    return a, b


def build_injection_filter(c1t: float, c1b: float, c2t: float, c2b: float,
                           c3: float, c4: float):
    """Build the noise injection filter (F2n).

    MAME neutralizes this filter (all coefficients zeroed).
    Returns zeroed (a, b) arrays of length 2.
    """
    a = np.zeros(2)
    b = np.array([1.0, 0.0])
    return a, b


def apply_filter(x_hist, y_hist, a, b):
    """Apply IIR filter: compute one output sample.

    x_hist[0] is the current input x[n], x_hist[1] is x[n-1], etc.
    y_hist[0] is y[n-1], y_hist[1] is y[n-2], etc.
    b[0] is assumed to be 1.0 (normalized).

    Returns the new output sample.
    """
    result = 0.0
    na = min(len(a), len(x_hist))
    for i in range(na):
        result += a[i] * x_hist[i]

    nb = min(len(b), len(y_hist) + 1)
    for i in range(1, nb):
        result -= b[i] * y_hist[i - 1]

    return result


def shift_hist(val: float, hist: np.ndarray):
    """Push a new value into a history buffer, shifting old values right."""
    for i in range(len(hist) - 1, 0, -1):
        hist[i] = hist[i - 1]
    if len(hist) > 0:
        hist[0] = val
