"""Tests for filter construction and stability."""

import math
import numpy as np
import pytest
from pyvotrax.filters import (
    bits_to_caps, build_standard_filter, build_noise_shaper_filter,
    build_lowpass_filter, build_injection_filter, apply_filter, shift_hist,
    SCLOCK, CCLOCK, MASTER_CLOCK,
)


class TestClockConstants:
    def test_master_clock(self):
        assert MASTER_CLOCK == 720_000

    def test_sclock(self):
        assert SCLOCK == 40_000

    def test_cclock(self):
        assert CCLOCK == 20_000


class TestBitsToCaps:
    def test_zero_value(self):
        assert bits_to_caps(0, (100, 200, 400, 800)) == 0.0

    def test_all_bits_set(self):
        assert bits_to_caps(0xF, (100, 200, 400, 800)) == 1500.0

    def test_single_bits(self):
        caps = (10, 20, 40, 80)
        assert bits_to_caps(1, caps) == 10.0
        assert bits_to_caps(2, caps) == 20.0
        assert bits_to_caps(4, caps) == 40.0
        assert bits_to_caps(8, caps) == 80.0

    def test_mixed_bits(self):
        caps = (100, 200, 400, 800)
        assert bits_to_caps(0b0101, caps) == 500.0   # 100 + 400
        assert bits_to_caps(0b1010, caps) == 1000.0  # 200 + 800

    def test_5bit_caps(self):
        """F2 uses 5-bit caps."""
        caps = (833, 1663, 3164, 6327, 12654)
        assert bits_to_caps(0b11111, caps) == sum(caps)
        assert bits_to_caps(0b00001, caps) == 833


class TestStandardFilter:
    def test_f1_produces_finite_coefficients(self):
        """F1 filter with typical parameters."""
        a, b = build_standard_filter(11247, 11797, 949, 52067, 2280 + 2546, 166272)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))
        assert len(a) == 4
        assert len(b) == 4

    def test_b0_is_normalized(self):
        """After normalization, b[0] should be 1.0."""
        a, b = build_standard_filter(11247, 11797, 949, 52067, 2280 + 2546, 166272)
        assert b[0] == pytest.approx(1.0)

    def test_f2v_produces_finite_coefficients(self):
        """F2v filter with typical parameters."""
        a, b = build_standard_filter(24840, 29154, 829 + 1390, 38180, 2352 + 833, 34270)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))

    def test_f3_produces_finite_coefficients(self):
        """F3 filter with typical parameters."""
        a, b = build_standard_filter(0, 17594, 868, 18828, 8480 + 2226, 50019)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))

    def test_f4_fixed_produces_finite_coefficients(self):
        """F4 fixed filter."""
        a, b = build_standard_filter(0, 28810, 1165, 21457, 8558, 7289)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))

    def test_3rd_order_coefficients_nonzero(self):
        """MAME standard filters are 3rd order — all 4 coefficients should be nonzero."""
        a, b = build_standard_filter(11247, 11797, 949, 52067, 2280 + 2546, 166272)
        # a[3] and b[3] should be nonzero for a true 3rd-order filter
        assert a[3] != 0.0
        assert b[3] != 0.0

    def test_poles_inside_unit_circle(self):
        """All standard filter poles should be inside the unit circle (stable)."""
        test_cases = [
            (11247, 11797, 949, 52067, 2280 + 9861, 166272),  # F1
            (24840, 29154, 829 + 5875, 38180, 2352 + 3164, 34270),  # F2v
            (0, 17594, 868, 18828, 8480 + 9056, 50019),  # F3
            (0, 28810, 1165, 21457, 8558, 7289),  # F4
        ]
        for params in test_cases:
            a, b = build_standard_filter(*params)
            poles = np.roots(b)
            for pole in poles:
                assert abs(pole) < 1.0 + 1e-6, \
                    f"Unstable pole {pole} (mag={abs(pole)}) for params {params}"


class TestNoiseShaper:
    def test_produces_finite_coefficients(self):
        a, b = build_noise_shaper_filter(15500, 14854, 8450, 9523, 14083)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))
        assert len(a) == 3
        assert len(b) == 3

    def test_b0_is_normalized(self):
        a, b = build_noise_shaper_filter(15500, 14854, 8450, 9523, 14083)
        assert b[0] == pytest.approx(1.0)

    def test_bandpass_structure(self):
        """Noise shaper should have a[1]=0 (bandpass zero at DC)."""
        a, b = build_noise_shaper_filter(15500, 14854, 8450, 9523, 14083)
        assert a[1] == pytest.approx(0.0)

    def test_antisymmetric_a(self):
        """a[0] and a[2] should be equal magnitude, opposite sign."""
        a, b = build_noise_shaper_filter(15500, 14854, 8450, 9523, 14083)
        assert a[0] == pytest.approx(-a[2])


class TestLowpassFilter:
    def test_produces_finite_coefficients(self):
        a, b = build_lowpass_filter(1122, 23131)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))
        assert len(a) == 2
        assert len(b) == 2

    def test_b0_is_normalized(self):
        a, b = build_lowpass_filter(1122, 23131)
        assert b[0] == pytest.approx(1.0)

    def test_dc_gain(self):
        """MAME lowpass DC gain is ~0.5 (single a[0] coeff, no a[1] term)."""
        a, b = build_lowpass_filter(1122, 23131)
        dc_gain = np.sum(a) / np.sum(b)
        assert abs(dc_gain - 0.5) < 0.05, f"DC gain = {dc_gain}, expected ~0.5"

    def test_pole_inside_unit_circle(self):
        a, b = build_lowpass_filter(1122, 23131)
        poles = np.roots(b)
        for pole in poles:
            assert abs(pole) < 1.0 + 1e-10


class TestInjectionFilter:
    # Typical cap values for F2n (from SH phoneme region)
    CAPS = (29154, 829, 38180, 2352, 34270)

    def test_highpass_coefficient_signs(self):
        """IIR should have a[0] > 0 and a[1] < 0 (highpass character)."""
        a, b = build_injection_filter(*self.CAPS)
        assert a[0] > 0
        assert a[1] < 0

    def test_stable_pole(self):
        """IIR pole should be inside the unit circle (stable)."""
        a, b = build_injection_filter(*self.CAPS)
        assert b[0] == pytest.approx(1.0)
        pole = -b[1]
        assert 0 < pole < 1.0, f"Pole at z={pole}, expected 0 < pole < 1"

    def test_dc_gain_matches_analog(self):
        """DC gain should equal k0/k1 (small but nonzero)."""
        c1b, c2t, c2b, c3, c4 = self.CAPS
        a, b = build_injection_filter(*self.CAPS)

        k0 = c2t / (CCLOCK * c1b)
        k1 = c4 * c2t / (CCLOCK * c1b * c3)
        expected_dc = k0 / k1

        dc_gain = np.sum(a) / np.sum(b)
        assert dc_gain == pytest.approx(expected_dc, rel=1e-6)
        assert 0 < dc_gain < 0.5  # small but nonzero

    def test_gain_increases_with_frequency(self):
        """Higher frequencies should have higher gain (highpass behavior)."""
        a, b = build_injection_filter(*self.CAPS)
        gains = []
        for f in [1000, 5000, 10000]:
            w = 2 * math.pi * f / SCLOCK
            # H(e^jw) for IIR: (a[0] + a[1]*e^(-jw)) / (1 + b[1]*e^(-jw))
            ejw = np.exp(-1j * w)
            h = (a[0] + a[1] * ejw) / (1.0 + b[1] * ejw)
            gains.append(abs(h))
        assert gains[0] < gains[1] < gains[2]

    def test_guard_clause_zeroes_filter(self):
        """When k1 <= 0 (e.g. c4=0), filter should be zeroed."""
        a, b = build_injection_filter(29154, 829, 38180, 2352, 0)
        assert np.all(a == 0.0)
        assert b[0] == pytest.approx(1.0)

    def test_matches_analog_at_multiple_freqs(self):
        """IIR gain should match the exact analog magnitude at 1k, 5k, 10k Hz."""
        c1b, c2t, c2b, c3, c4 = self.CAPS
        a, b = build_injection_filter(*self.CAPS)

        k0 = c2t / (CCLOCK * c1b)
        k1 = c4 * c2t / (CCLOCK * c1b * c3)
        k2 = c4 * c2b / (CCLOCK * CCLOCK * c1b * c3)

        for f in [1000, 5000, 10000]:
            w = 2 * math.pi * f
            # Exact analog: |H(jw)| = |(k0 + k2*jw)| / |(k1 + k2*jw)|
            analog_gain = abs(k0 + 1j * k2 * w) / abs(k1 + 1j * k2 * w)

            # Digital IIR gain
            wd = 2 * math.pi * f / SCLOCK
            ejw = np.exp(-1j * wd)
            digital_gain = abs((a[0] + a[1] * ejw) / (1.0 + b[1] * ejw))

            assert digital_gain == pytest.approx(analog_gain, rel=0.05), \
                f"Mismatch at {f} Hz: digital={digital_gain:.4f}, analog={analog_gain:.4f}"

    def test_nonzero_output_for_varying_input(self):
        """IIR should produce nonzero output for a changing input signal."""
        a, b = build_injection_filter(*self.CAPS)
        x_hist = np.array([1.0, 0.0])
        y_hist = np.array([0.0])
        result = apply_filter(x_hist, y_hist, a, b)
        assert result != 0.0

    def test_dc_output_small(self):
        """DC output should be small but nonzero (DC gain = k0/k1)."""
        a, b = build_injection_filter(*self.CAPS)
        # Run filter with DC input until it converges
        x_hist = np.array([1.0, 1.0])
        y_hist = np.array([0.0])
        for _ in range(100):
            result = apply_filter(x_hist, y_hist, a, b)
            y_hist[0] = result
        # Steady-state output should be small (DC gain < 0.5)
        assert abs(result) < 0.5
        assert abs(result) > 0.0


class TestApplyFilter:
    def test_simple_gain(self):
        """A filter with a[0]=2 and no feedback should double the input."""
        a = np.array([2.0])
        b = np.array([1.0])
        x_hist = np.array([3.0])
        y_hist = np.array([])
        result = apply_filter(x_hist, y_hist, a, b)
        assert result == 6.0

    def test_feedback(self):
        """Test that feedback (b[1]) is applied correctly."""
        a = np.array([1.0])
        b = np.array([1.0, -0.5])
        x_hist = np.array([1.0])
        y_hist = np.array([2.0])  # y[n-1] = 2.0
        result = apply_filter(x_hist, y_hist, a, b)
        # result = 1.0*1.0 - (-0.5)*2.0 = 1.0 + 1.0 = 2.0
        assert result == 2.0


class TestShiftHist:
    def test_shift(self):
        hist = np.array([1.0, 2.0, 3.0])
        shift_hist(5.0, hist)
        np.testing.assert_array_equal(hist, [5.0, 1.0, 2.0])

    def test_single_element(self):
        hist = np.array([1.0])
        shift_hist(5.0, hist)
        np.testing.assert_array_equal(hist, [5.0])
