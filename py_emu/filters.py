"""Votrax SC-01A analog filter construction using bilinear z-transform.

All filter designs and capacitor values from MAME votrax.cpp die analysis.
"""

import math
import numpy as np

# Clock constants
MASTER_CLOCK = 720_000
SCLOCK = MASTER_CLOCK // 18   # 40000 Hz - analog sample rate
CCLOCK = SCLOCK // 2          # 20000 Hz - chip update rate


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


def build_standard_filter(c1t, c1b, c2t, c2b, c3, c4):
    """Build a 3rd-order formant filter (F1, F2v, F3, F4).

    H(s) = (1 + k0*s) / (1 + k1*s + k2*s^2)

    Returns (a, b) as numpy arrays of length 4, matching MAME's
    bilinear transform with frequency pre-warping (votrax.cpp:849-884).
    """
    # Coefficients from circuit analysis (cap ratios cancel physical units)
    k0 = c1t / (CCLOCK * c1b)
    k1 = c4 * c2t / (CCLOCK * c1b * c3)
    k2 = c4 * c2b / (CCLOCK * CCLOCK * c1b * c3)

    # Estimate the filter cutoff frequency
    fpeak = math.sqrt(abs(k0 * k1 - k2)) / (2 * math.pi * k2)

    # Turn that into a warp multiplier
    zc = 2 * math.pi * fpeak / math.tan(math.pi * fpeak / SCLOCK)

    # Z-transform coefficients
    m0 = zc * k0
    m1 = zc * k1
    m2 = zc * zc * k2

    a = np.array([1 + m0, 3 + m0, 3 - m0, 1 - m0])
    b = np.array([1 + m1 + m2, 3 + m1 - m2, 3 - m1 - m2, 1 - m1 + m2])

    # Normalize so b[0]=1 (apply_filter assumes this)
    a /= b[0]
    b /= b[0]
    return a, b


def build_noise_shaper_filter(c1, c2t, c2b, c3, c4):
    """Build the noise shaper filter (2nd-order bandpass).

    H(s) = k0*s / (1 + k1*s + k2*s^2)

    Returns (a, b) numpy arrays of length 3, matching MAME's
    bilinear transform (votrax.cpp:957-986).
    """
    # Coefficients from circuit analysis
    k0 = c2t * c3 * c2b / c4
    k1 = c2t * (CCLOCK * c2b)
    k2 = c1 * c2t * c3 / (CCLOCK * c4)

    # Estimate the filter cutoff frequency
    fpeak = math.sqrt(1 / k2) / (2 * math.pi)

    # Turn that into a warp multiplier
    zc = 2 * math.pi * fpeak / math.tan(math.pi * fpeak / SCLOCK)

    # Z-transform coefficients
    m0 = zc * k0
    m1 = zc * k1
    m2 = zc * zc * k2

    a = np.array([m0, 0.0, -m0])
    b = np.array([1 + m1 + m2, 2 - 2 * m2, 1 - m1 + m2])

    # Normalize so b[0]=1
    a /= b[0]
    b /= b[0]
    return a, b


def build_lowpass_filter(c1t, c1b):
    """Build the final output lowpass filter (FX).

    MAME: "The caps values puts the cutoff at around 150Hz,
    but that's no good. Recordings shows we want it around 4K, so fuzz it."

    Returns (a, b) numpy arrays of length 2, matching MAME's
    bilinear transform (votrax.cpp:904-926).
    """
    # Compute the coefficient with MAME's 150/4000 fudge factor
    k = c1b / (CCLOCK * c1t) * (150.0 / 4000.0)

    # Compute the filter cutoff frequency
    fpeak = 1 / (2 * math.pi * k)

    # Turn that into a warp multiplier
    zc = 2 * math.pi * fpeak / math.tan(math.pi * fpeak / SCLOCK)

    # Z-transform coefficient
    m = zc * k

    # MAME: a[0]=1, b[0]=1+m, b[1]=1-m  (1st order, single a coeff)
    a = np.array([1.0 / (1 + m), 0.0])
    b = np.array([1.0, (1 - m) / (1 + m)])
    return a, b


def build_injection_filter(c1b, c2t, c2b, c3, c4):
    """Build the noise injection filter (F2n) via pole-reflected bilinear transform.

    The analog circuit: H(s) = (k0 + k2*s) / (k1 - k2*s) has a RHP pole (unstable).
    Reflecting the pole: H_stable(s) = (k0 + k2*s) / (k1 + k2*s) preserves the
    magnitude response while placing the pole in the LHP (stable). Bilinear transform
    then yields a first-order IIR with the pole guaranteed inside the unit circle.
    """
    k0 = c2t / (CCLOCK * c1b)
    k1 = c4 * c2t / (CCLOCK * c1b * c3)
    k2 = c4 * c2b / (CCLOCK * CCLOCK * c1b * c3)

    if k1 <= 0:
        return np.zeros(2), np.array([1.0, 0.0])

    c = 2.0 * SCLOCK  # bilinear transform constant
    denom = k1 + k2 * c

    a = np.array([(k0 + k2 * c) / denom, (k0 - k2 * c) / denom])
    b = np.array([1.0, (k1 - k2 * c) / denom])
    return a, b


def build_enhanced_noise_filter(center_freq, bandwidth):
    """Build a 2nd-order bandpass filter for enhanced per-fricative noise shaping.

    Uses bilinear transform of analog bandpass. Gain is matched to produce
    output levels comparable to the standard noise shaper, accounting for
    the bandpass's resonant energy accumulation.

    Args:
        center_freq: Center frequency in Hz.
        bandwidth: Bandwidth in Hz.

    Returns:
        (a, b) numpy arrays of length 3, normalized so b[0]=1.
    """
    alpha = math.pi * bandwidth / SCLOCK
    cos_w0 = math.cos(2 * math.pi * center_freq / SCLOCK)

    a_arr = np.array([alpha, 0.0, -alpha])
    b_arr = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])

    # Normalize b[0]=1
    a_arr /= b_arr[0]
    b_arr /= b_arr[0]

    # Scale gain to match standard noise shaper output level.
    target_gain = 1e-8
    a_arr *= target_gain

    return a_arr, b_arr


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
