"""Constant-pitch rate control.

Tempo and pitch come from different places on this chip. Pitch is the glottal
oscillator, clocked straight off the master clock; tempo is the duration
counter, which only decides when the chip asks for the next phone. Commit the
next phone early and the speech speeds up with the pitch untouched.

That works only because a phone's natural length is exactly predictable, so
these tests pin the formula down. The scheduler that uses it lives in
csrc/votrax_capi.cpp and is exercised through the C API; what is testable from
Python is the arithmetic it rests on.
"""

import numpy as np
import pytest

from py_emu.rom import ROM_DATA
from pyvotrax._votrax_core import VotraxSC01ACore
from pyvotrax.chip import MaskRevision, VotraxSC01A
from pyvotrax.phonemes import PHONE_TABLE, name_to_code


class TestPhoneLength:
    """A phone runs 16 ticks; a tick is (4*duration + 1) chip updates; a chip
    update is two samples. Hence 32 * (4*duration + 1), exactly."""

    @pytest.mark.parametrize("phone", range(64))
    def test_formula_matches_the_chip(self, phone):
        chip = VotraxSC01A()
        chip.reset()
        for _ in range(64):
            chip.generate_one_sample()      # settle the reset
        chip.phone_commit(phone, 0)
        measured = 0
        while not chip.phone_done and measured < 400_000:
            chip.generate_one_sample()
            measured += 1
        assert measured == 32 * (4 * ROM_DATA[phone].duration + 1)

    @pytest.mark.parametrize("phone", range(64))
    def test_core_reports_the_same(self, phone):
        core = VotraxSC01ACore()
        assert core.phone_samples(phone) == 32 * (4 * ROM_DATA[phone].duration + 1)

    def test_phone_length_does_not_vary_with_mask(self):
        """duration lives in word0, which is identical between the masks."""
        a = VotraxSC01ACore(mask=MaskRevision.SC01A)
        b = VotraxSC01ACore(mask=MaskRevision.SC01)
        for phone in range(64):
            assert a.phone_samples(phone) == b.phone_samples(phone)

    def test_lengths_are_sane(self):
        core = VotraxSC01ACore()
        lengths = [core.phone_samples(p) for p in range(64)]
        assert all(n > 0 for n in lengths)
        # 40 kHz: the shortest phone is about 49 ms, the longest about 244 ms
        assert min(lengths) / 40_000 == pytest.approx(0.0488, abs=1e-3)
        assert max(lengths) / 40_000 == pytest.approx(0.2440, abs=1e-3)

    def test_phone_length_is_masked_to_six_bits(self):
        core = VotraxSC01ACore()
        assert core.phone_samples(0x7F) == core.phone_samples(0x3F)


class TestTruncationChangesTempoNotPitch:
    """The property the whole scheme rests on: the glottal period is a
    function of the clock alone, so truncating phones cannot move it."""

    SEQ = ["AH", "L", "AH", "L", "AH", "L", "AH"]

    def _render(self, speed):
        """The scheduler, in miniature: hold each phone natural/speed samples."""
        chip = VotraxSC01A()
        chip.reset()
        out = []
        for name in self.SEQ:
            phone = name_to_code(name)
            hold = max(1, round(chip._native.phone_samples(phone) / speed))
            chip.phone_commit(phone, 0)
            for _ in range(hold):
                out.append(chip.generate_one_sample())
        return np.array(out)

    def _period(self, signal, rate=40_000):
        """Glottal period by autocorrelation, in samples."""
        sig = signal - signal.mean()
        lo, hi = rate // 300, rate // 60
        corr = [float(np.dot(sig[:-lag], sig[lag:])) for lag in range(lo, hi)]
        return lo + int(np.argmax(corr))

    def test_tempo_scales_with_speed(self):
        natural = len(self._render(1.0))
        for speed in (0.5, 1.5, 2.0, 3.0):
            got = len(self._render(speed))
            assert got == pytest.approx(natural / speed, rel=0.01)

    @pytest.mark.parametrize("speed", [0.7, 1.5, 2.0, 3.0])
    def test_pitch_does_not_move(self, speed):
        base = self._period(self._render(1.0))
        fast = self._period(self._render(speed))
        # Truncating shortens the signal, so the autocorrelation estimate
        # gets noisier as speed rises: a few samples out of ~510. 2% is well
        # inside that noise, and nowhere near the 100% that clock scaling
        # produces -- which is the comparison the next test makes.
        assert base == pytest.approx(fast, rel=0.02)

    def test_clock_scaling_does_move_pitch(self):
        """The contrast: the other way of going faster is not pitch-safe."""
        slow = VotraxSC01A(master_clock=720_000)
        fast = VotraxSC01A(master_clock=1_440_000)
        seq = [name_to_code(n) for n in self.SEQ]
        out = []
        for chip in (slow, fast):
            chip.reset()
            samples = []
            for phone in seq:
                chip.phone_commit(phone, 0)
                for _ in range(chip._native.phone_samples(phone)):
                    samples.append(chip.generate_one_sample())
            out.append(np.array(samples))
        # Same sample count, but the second is at twice the sample rate, so it
        # is half the wall-clock duration — and the period in samples is equal,
        # which at double the rate means double the frequency.
        assert len(out[0]) == len(out[1])
        assert self._period(out[0]) == pytest.approx(self._period(out[1]), abs=2)
