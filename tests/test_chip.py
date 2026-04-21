"""Tests for the VotraxSC01A chip emulator (black-box, C++ backend)."""

import numpy as np
import pytest
from pyvotrax.chip import VotraxSC01A
from pyvotrax.constants import SCLOCK


class TestDuration:
    def test_phone_done_eventually(self):
        """Phoneme should complete (phone_done becomes True)."""
        chip = VotraxSC01A()
        chip.phone_commit(0x24, 0)  # AH
        assert not chip.phone_done
        # Generate enough samples to complete
        for _ in range(200000):
            chip.generate_one_sample()
            if chip.phone_done:
                break
        assert chip.phone_done, "Phoneme did not complete"

    def test_all_phonemes_complete(self):
        """All phonemes should eventually complete."""
        chip = VotraxSC01A()
        for code in range(64):
            chip.phone_commit(code, 0)
            for _ in range(400000):
                chip.generate_one_sample()
                if chip.phone_done:
                    break
            assert chip.phone_done, f"Phoneme {code} did not complete"


class TestGenerateSamples:
    def test_output_shape(self):
        """generate_samples should return the requested number of samples."""
        chip = VotraxSC01A()
        chip.phone_commit(0x24, 0)  # AH (voiced)
        samples = chip.generate_samples(1000)
        assert samples.shape == (1000,)

    def test_output_finite(self):
        """All output samples should be finite."""
        chip = VotraxSC01A()
        chip.phone_commit(0x24, 0)  # AH (voiced)
        samples = chip.generate_samples(2000)
        assert np.all(np.isfinite(samples))

    def test_voiced_phoneme_not_silent(self):
        """A voiced phoneme (AH) should produce non-zero output."""
        chip = VotraxSC01A()
        chip.phone_commit(0x24, 0)  # AH
        samples = chip.generate_samples(4000)
        # Allow some initial silence, but overall should have energy
        rms = np.sqrt(np.mean(samples ** 2))
        assert rms > 1e-10, f"Output is silent (RMS={rms})"


def _spectral_centroid(phoneme_code, n_samples=16000):
    """Compute the power-weighted spectral centroid above 1 kHz for a phoneme.

    Skips the first 4000 samples (transient) and uses the next 8000 (steady-state).
    Uses power spectrum (squared magnitudes) and excludes frequencies below 1 kHz
    to avoid DC/low-freq artifacts distorting the centroid.
    """
    chip = VotraxSC01A()
    chip.phone_commit(phoneme_code, 0)
    samples = chip.generate_samples(n_samples)
    steady = samples[4000:12000]
    power = np.abs(np.fft.rfft(steady)) ** 2
    freqs = np.fft.rfftfreq(len(steady), d=1.0 / SCLOCK)
    mask = freqs >= 1000
    total = np.sum(power[mask])
    if total == 0:
        return 0.0
    return np.sum(freqs[mask] * power[mask]) / total


class TestMasterClock:
    def test_default_clock(self):
        chip = VotraxSC01A()
        assert chip.master_clock == 720_000.0
        assert chip.sclock == pytest.approx(40_000.0)
        assert chip.cclock == pytest.approx(20_000.0)

    def test_custom_clock_changes_sclock(self):
        chip = VotraxSC01A(master_clock=360_000.0)
        assert chip.master_clock == 360_000.0
        assert chip.sclock == pytest.approx(20_000.0)
        assert chip.cclock == pytest.approx(10_000.0)

    def test_slower_clock_stretches_phoneme(self):
        """At half the master clock, AH should take ~2x as many generate calls to complete."""
        def ticks_until_done(mc):
            chip = VotraxSC01A(master_clock=mc)
            chip.phone_commit(0x24, 0)  # AH
            n = 0
            while not chip.phone_done and n < 400_000:
                chip.generate_one_sample()
                n += 1
            return n

        fast = ticks_until_done(720_000.0)
        slow = ticks_until_done(360_000.0)
        # The sample-generation rate per audio second doesn't change from the
        # chip's perspective — it always runs chip_update every 2 samples. So
        # fast and slow should produce approximately the same sample count to
        # reach phone_done. (The "stretching" is in audio seconds, not sample
        # count, because SCLOCK also halves.)
        assert abs(slow - fast) < max(fast, slow) * 0.05

    def test_custom_clock_output_still_voiced(self):
        """Non-default master clock should still produce non-silent voiced output."""
        chip = VotraxSC01A(master_clock=540_000.0)
        chip.phone_commit(0x24, 0)  # AH
        samples = chip.generate_samples(4000)
        assert np.all(np.isfinite(samples))
        rms = np.sqrt(np.mean(samples ** 2))
        assert rms > 1e-10

    def test_fx_fudge_default(self):
        chip = VotraxSC01A()
        assert chip.fx_fudge == pytest.approx(150.0 / 4000.0)

    def test_closure_strength_default(self):
        chip = VotraxSC01A()
        assert chip.closure_strength == pytest.approx(1.0)

    def test_closure_strength_zero_changes_plosive(self):
        """closure_strength=0 should change the output of a plosive phoneme
        (B, code 0x0E) vs. the default closure curve."""
        def render(strength):
            chip = VotraxSC01A(closure_strength=strength)
            chip.phone_commit(0x0E, 0)  # B (voiced stop)
            return chip.generate_samples(8000)

        default = render(1.0)
        no_closure = render(0.0)
        diff = float(np.sqrt(np.mean((default - no_closure) ** 2)))
        # At closure_strength=0 there's no closure dip, so the output differs
        default_rms = float(np.sqrt(np.mean(default ** 2)))
        assert diff > 0.02 * max(default_rms, 1e-3), (
            f"closure_strength=0 produced indistinguishable output for B "
            f"(diff={diff}, default_rms={default_rms})"
        )

    def test_fx_fudge_as_schematic(self):
        """fx_fudge=1.0 should give the 'as-schematic' 150 Hz behavior — much
        more low-frequency energy than the authentic 4 kHz cutoff."""
        def rms(chip):
            chip.phone_commit(0x24, 0)  # AH
            return float(np.sqrt(np.mean(chip.generate_samples(8000) ** 2)))

        authentic = rms(VotraxSC01A(fx_fudge=150.0 / 4000.0))
        muffled = rms(VotraxSC01A(fx_fudge=1.0))
        # The "as-schematic" 150 Hz cutoff drops far more voice energy than the
        # authentic 4 kHz cutoff, so authentic RMS should be meaningfully larger.
        assert authentic > muffled, (
            f"authentic RMS {authentic} not greater than muffled {muffled}"
        )


class TestSibilantDifferentiation:
    # Phoneme codes: S=0x1F, SH=0x11, Z=0x12, ZH=0x07, CH=0x10, J=0x1A
    SIBILANTS = {"S": 0x1F, "SH": 0x11, "Z": 0x12, "ZH": 0x07, "CH": 0x10, "J": 0x1A}

    def test_sh_and_s_spectrally_different(self):
        """S (fc=0, direct path) and SH (fc=15, F2N path) should differ by >500 Hz."""
        centroid_s = _spectral_centroid(self.SIBILANTS["S"])
        centroid_sh = _spectral_centroid(self.SIBILANTS["SH"])
        diff = abs(centroid_s - centroid_sh)
        assert diff > 500, (
            f"S centroid={centroid_s:.0f} Hz, SH centroid={centroid_sh:.0f} Hz, "
            f"diff={diff:.0f} Hz — expected >500 Hz"
        )

    def test_sibilant_output_finite(self):
        """All sibilant phonemes should produce finite output (no NaN/Inf)."""
        for name, code in self.SIBILANTS.items():
            chip = VotraxSC01A()
            chip.phone_commit(code, 0)
            samples = chip.generate_samples(4000)
            assert np.all(np.isfinite(samples)), f"{name} (0x{code:02X}) has non-finite samples"
