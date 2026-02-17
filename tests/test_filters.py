"""Tests for filter construction and stability."""

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
            # Get poles from denominator polynomial
            if b[0] != 0:
                poles = np.roots(b[:3])  # Use first 3 coefficients (2nd order)
                for pole in poles:
                    assert abs(pole) < 1.0 + 1e-10, \
                        f"Unstable pole {pole} (mag={abs(pole)}) for params {params}"

    def test_zero_c3_returns_zeros(self):
        a, b = build_standard_filter(100, 100, 100, 100, 0, 100)
        assert a[0] == 0.0


class TestNoiseShaper:
    def test_produces_finite_coefficients(self):
        a, b = build_noise_shaper_filter(15500, 14854, 8450, 9523, 14083)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))
        assert len(a) == 3
        assert len(b) == 3

    def test_poles_inside_unit_circle(self):
        a, b = build_noise_shaper_filter(15500, 14854, 8450, 9523, 14083)
        poles = np.roots(b)
        for pole in poles:
            assert abs(pole) < 1.0 + 1e-10, f"Unstable pole {pole}"


class TestLowpassFilter:
    def test_produces_finite_coefficients(self):
        a, b = build_lowpass_filter(1122, 23131)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))
        assert len(a) == 2
        assert len(b) == 2

    def test_unity_dc_gain(self):
        """Lowpass should pass DC (sum of a / sum of b ≈ 1 or known gain)."""
        a, b = build_lowpass_filter(1122, 23131)
        dc_gain = np.sum(a) / np.sum(b)
        # DC gain should be close to 1 for a normalized lowpass
        assert abs(dc_gain - 1.0) < 0.01, f"DC gain = {dc_gain}"

    def test_pole_inside_unit_circle(self):
        a, b = build_lowpass_filter(1122, 23131)
        poles = np.roots(b)
        for pole in poles:
            assert abs(pole) < 1.0 + 1e-10


class TestInjectionFilter:
    def test_returns_zeroed_a(self):
        a, b = build_injection_filter(0, 0, 0, 0, 0, 0)
        assert np.all(a == 0.0)

    def test_produces_zero_output(self):
        """Injection filter should produce zero output for any input."""
        a, b = build_injection_filter(0, 0, 0, 0, 0, 0)
        x_hist = np.array([1.0, 0.5])
        y_hist = np.array([0.0])
        result = apply_filter(x_hist, y_hist, a, b)
        assert result == 0.0


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
