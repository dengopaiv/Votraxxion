"""The synthesizer against the Votrax SC-01 data sheet (1980).

`reference/SC-01_Data_Sheet_v1_text.pdf` is the manufacturer's statement of
how the chip behaves, and the one source here that owes nothing to MAME or to
the die photographs. `docs/DATASHEET.md` lists what it says and how each point
was checked; this module is where the checks live.

Skipped unless the library has been built (nvda-addon/package.py). The ROM
category test reads `py_emu.rom`, whose tables `tools/verify_rom.py` proves
identical to the C ones and to the dumped masks.
"""

import ctypes
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

DLL = ROOT / "nvda-addon" / "addon" / "synthDrivers" / (
    "votraxNative-x64.dll" if ctypes.sizeof(ctypes.c_void_p) == 8
    else "votraxNative-x86.dll"
)

pytestmark = pytest.mark.skipif(
    not DLL.is_file(),
    reason=f"{DLL.name} not built - run nvda-addon/package.py",
)

#: Table 1, "Phoneme Chart": code order, symbol, duration in ms at 720 kHz.
TABLE_1 = [
    ("EH3", 59), ("EH2", 71), ("EH1", 121), ("PA0", 47), ("DT", 47),
    ("A2", 71), ("A1", 103), ("ZH", 90), ("AH2", 71), ("I3", 55),
    ("I2", 80), ("I1", 121), ("M", 103), ("N", 80), ("B", 71), ("V", 71),
    ("CH", 71), ("SH", 121), ("Z", 71), ("AW1", 146), ("NG", 121),
    ("AH1", 146), ("OO1", 103), ("OO", 185), ("L", 103), ("K", 80),
    ("J", 47), ("H", 71), ("G", 71), ("F", 103), ("D", 55), ("S", 90),
    ("A", 185), ("AY", 65), ("Y1", 80), ("UH3", 47), ("AH", 250), ("P", 103),
    ("O", 185), ("I", 185), ("U", 185), ("Y", 103), ("T", 71), ("R", 90),
    ("E", 185), ("W", 80), ("AE", 185), ("AE1", 103), ("AW2", 90),
    ("UH2", 71), ("UH1", 103), ("UH", 185), ("O2", 80), ("O1", 121),
    ("IU", 59), ("U1", 90), ("THV", 80), ("TH", 71), ("ER", 146),
    ("EH", 185), ("E1", 121), ("AW", 250), ("PA1", 185), ("STOP", 47),
]

#: Table 2, "Phoneme Categories According to Production Features".
TABLE_2 = {
    "voiced": "E E1 Y Y1 I I1 I2 I3 EH EH1 EH2 EH3 A A1 A2 AY AE AE1 AH AH1 "
              "AH2 AW AW1 AW2 UH UH1 UH2 UH3 O O1 O2 OO OO1 R ER L IU U U1 W",
    "voiced fricative": "Z ZH J V THV",
    "voiced stop": "B D G",
    "fricative stop": "T DT K P",
    "fricative": "S SH CH TH F H",
    "nasal": "M N NG",
    "no sound": "PA0 PA1 STOP",
}


@pytest.fixture(scope="module")
def lib():
    import goldens
    return goldens.bind(str(DLL))


@pytest.fixture
def chip(lib):
    handle = lib.vx_create(0, 0)
    yield handle
    lib.vx_destroy(handle)


def render(lib, chip, n):
    buf = (ctypes.c_int16 * n)()
    lib.vx_render(chip, buf, n)
    return np.frombuffer(bytes(memoryview(buf).cast("B")), dtype=np.int16).astype(float)


def pitch_hz(samples, rate, lo=60.0, hi=160.0):
    """Fundamental by autocorrelation over the SC-01's whole pitch range."""
    x = samples - samples.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    lags = np.arange(int(rate / hi), int(rate / lo))
    return rate / lags[np.argmax(ac[lags])]


# --------------------------------------------------------------- Table 1 ---

class TestPhonemeChart:
    def test_codes_and_symbols(self, lib):
        names = [lib.vx_phone_name(i).decode() for i in range(64)]
        assert names == [name for name, _ in TABLE_1]

    def test_every_duration_is_within_seven_percent(self, lib, chip):
        """The chart's milliseconds against the exact phone length.

        Ours run a mean 3% long, worst +6.2% (the 47 ms phones) and -2.4% (AH
        and AW, 244 ms against 250); no phone is off by more than 6 ms. The
        chart is rounded to the ms and the ROM counts in 16 ticks, so exact
        agreement is not on offer; seven percent catches a wrong table, a
        wrong tick length or a transposed pair.
        """
        rate = lib.vx_sample_rate(chip)
        assert rate == 40000.0
        for code, (name, sheet_ms) in enumerate(TABLE_1):
            ours_ms = 1000.0 * lib.vx_phone_samples(chip, code) / rate
            assert abs(ours_ms - sheet_ms) / sheet_ms < 0.07, name
            assert abs(ours_ms - sheet_ms) <= 6.5, name

    def test_duration_range_matches_table_3(self, lib, chip):
        """Table 3: 'Time of Phoneme Duration' 47 min, 250 max, at 720 kHz."""
        rate = lib.vx_sample_rate(chip)
        ms = [1000.0 * lib.vx_phone_samples(chip, c) / rate for c in range(64)]
        assert 46 <= min(ms) <= 50
        assert 240 <= max(ms) <= 252

    def test_lower_clock_lengthens_phonemes(self, lib):
        """'As clock frequency decreases, audio frequency decreases and
        phoneme timing lengthens.'"""
        slow, fast = lib.vx_create(0, 360000), lib.vx_create(0, 1440000)
        try:
            n = lib.vx_phone_samples(slow, 0x24)
            assert n == lib.vx_phone_samples(fast, 0x24)
            assert n / lib.vx_sample_rate(slow) == pytest.approx(
                4 * n / lib.vx_sample_rate(fast))
        finally:
            lib.vx_destroy(slow)
            lib.vx_destroy(fast)


# --------------------------------------------------------------- Table 2 ---

class TestCategoriesInTheRom:
    """Table 2's categories are visible in the ROM's own fields.

    Voice amplitude (va), fricative amplitude (fa) and the closure flag
    separate six of the seven exactly, on both masks. Nasals cannot be told
    from vowels this way -- the SC-01 has no nasal anti-resonator, so M, N and
    NG are voiced sounds with low formants and nothing in the ROM marks them.
    """

    @pytest.fixture(params=["SC01A", "SC01"])
    def rom(self, request):
        from py_emu import rom as R
        table = R.ROM_DATA_SC01A if request.param == "SC01A" else R.ROM_DATA_SC01
        return {name: table[code] for code, (name, _) in enumerate(TABLE_1)}

    @staticmethod
    def members(category):
        return TABLE_2[category].split()

    def test_table_2_covers_all_64(self):
        everyone = sum((v.split() for v in TABLE_2.values()), [])
        assert sorted(everyone) == sorted(name for name, _ in TABLE_1)

    def test_voiced(self, rom):
        for n in self.members("voiced"):
            p = rom[n]
            assert p.va >= 8 and p.fa == 0 and p.closure == 0, n

    def test_voiced_fricative(self, rom):
        for n in self.members("voiced fricative"):
            p = rom[n]
            assert p.va == 1 and p.fa > 0 and p.closure == 0, n

    def test_voiced_stop(self, rom):
        for n in self.members("voiced stop"):
            p = rom[n]
            assert p.va > 0 and p.fa == 0 and p.closure == 1, n

    def test_fricative_stop(self, rom):
        for n in self.members("fricative stop"):
            p = rom[n]
            assert p.va == 0 and p.fa > 0 and p.closure == 1, n

    def test_fricative(self, rom):
        for n in self.members("fricative"):
            p = rom[n]
            assert p.va == 0 and p.fa > 0 and p.closure == 0, n

    def test_no_sound(self, rom):
        for n in self.members("no sound"):
            assert rom[n].va == 0 and rom[n].fa == 0, n

    def test_nasals_look_like_vowels(self, rom):
        for n in self.members("nasal"):
            p = rom[n]
            assert p.va > 0 and p.fa == 0 and p.closure == 0, n


# ------------------------------------------------- Table 1's footnotes ---

class TestAffricateRule:
    """"'T' must precede 'CH' to produce CH sound. 'D' must precede 'J' to
    produce J sound." The front end never emits either bare."""

    TEXT = ("church chip judge jam catch edge fudge nature question "
            "soldier gentle cheese jungle wretched")

    def _names(self, lib, fn, text):
        buf = (ctypes.c_ubyte * 8192)()
        n = fn(text.encode(), buf, 8192)
        return [lib.vx_phone_name(buf[i] & 0x3F).decode() for i in range(n)]

    @pytest.mark.parametrize("which", ["ttv_translate", "ttv_spell"])
    def test_no_bare_affricate(self, lib, which):
        names = self._names(lib, getattr(lib, which), self.TEXT + " h g j")
        assert "CH" in names and "J" in names
        for i, name in enumerate(names):
            if name == "CH":
                assert i > 0 and names[i - 1] == "T", names[max(0, i - 3):i + 1]
            if name == "J":
                assert i > 0 and names[i - 1] == "D", names[max(0, i - 3):i + 1]


# -------------------------------------------------- Signal Description ---

class TestInflectionPins:
    """I1, I2: 'Instantaneously sets pitch level of voiced phonemes.'"""

    def test_a_change_mid_phone_is_heard_in_that_phone(self, lib, chip):
        rate = lib.vx_sample_rate(chip)
        lib.vx_inflection(chip, 0)
        lib.vx_write(chip, 0x24)                  # AH, 244 ms
        render(lib, chip, 2400)                   # let the formants settle
        low = pitch_hz(render(lib, chip, 2400), rate)
        lib.vx_inflection(chip, 3)
        render(lib, chip, 400)                    # one glottal period to switch
        high = pitch_hz(render(lib, chip, 2400), rate)
        assert not lib.vx_ready(chip), "the phone ended; the test proves nothing"
        assert low < 85 and high > 115, (low, high)

    def test_the_four_levels_are_the_documented_pitches(self, lib):
        """votrax.h: about 78, 89, 104 and 125 Hz at the data sheet clock."""
        got = []
        for level in range(4):
            c = lib.vx_create(0, 0)
            lib.vx_inflection(c, level)
            lib.vx_write(c, 0x24)
            render(lib, c, 2400)
            got.append(pitch_hz(render(lib, c, 4000), lib.vx_sample_rate(c)))
            lib.vx_destroy(c)
        for measured, documented in zip(got, (78, 89, 104, 125)):
            assert measured == pytest.approx(documented, rel=0.04), got

    def test_a_contour_step_survives_a_change(self, lib, chip):
        """A phone queued a step above neutral stays a step above the new
        base: raising the base mid-phone from 1 to 2 lands it on 3."""
        rate = lib.vx_sample_rate(chip)
        lib.vx_speak(chip, (ctypes.c_ubyte * 1)(0x24 | (2 << 6)), 1)
        render(lib, chip, 2400)
        before = pitch_hz(render(lib, chip, 2400), rate)      # level 2
        lib.vx_inflection(chip, 2)
        render(lib, chip, 400)
        after = pitch_hz(render(lib, chip, 2400), rate)       # level 3
        assert before == pytest.approx(104, rel=0.05)
        assert after == pytest.approx(125, rel=0.05)


# ---------------------------------------------------------- Master clock ---

class TestMasterClock:
    def test_rc_relation(self, lib):
        """'Frequency of Master Clock ~ 1.25 / RC'."""
        assert lib.vx_clock_from_rc(6500.0, 300e-12) == round(1.25 / (6500 * 300e-12))
        assert lib.vx_clock_from_rc(0.0, 300e-12) == 0
        assert lib.vx_clock_from_rc(6500.0, -1.0) == 0

    def test_the_sheet_calls_its_own_typical_parts_approximate(self, lib):
        """6.5 k and 300 pF are the sheet's typical parts and 720 kHz its
        typical clock; the formula puts them 11% apart, which is the sheet's
        own ~, and why the relation is not used to set the default."""
        assert lib.vx_clock_from_rc(6500.0, 300e-12) == pytest.approx(641026, abs=1)

    def test_knob_ends_and_middle(self, lib):
        """Figure 8: 6.8 k + 50 k audio taper against 120 pF."""
        top = lib.vx_clock_from_knob(0.0)
        bottom = lib.vx_clock_from_knob(1.0)
        assert top == round(1.25 / (6800 * 120e-12))
        assert bottom == round(1.25 / (56800 * 120e-12))
        assert lib.vx_clock_from_knob(0.6) == pytest.approx(720000, rel=0.02)
        positions = [lib.vx_clock_from_knob(k / 20) for k in range(21)]
        assert positions == sorted(positions, reverse=True)
        assert lib.vx_clock_from_knob(-1) == top and lib.vx_clock_from_knob(2) == bottom

    def test_audio_taper_is_gentle_at_first(self, lib):
        """An audio taper gives 10% of its track at mid rotation, so half a
        turn moves the clock far less than the second half does."""
        top, mid, bottom = (lib.vx_clock_from_knob(p) for p in (0.0, 0.5, 1.0))
        assert (top - mid) < (mid - bottom)

    @staticmethod
    def _stream(lib, clock, phones):
        c = lib.vx_create(1, clock)
        lib.vx_speak(c, (ctypes.c_ubyte * len(phones))(*phones), len(phones))
        out = render(lib, c, 20000)
        lib.vx_destroy(c)
        return out

    def test_voiced_sound_is_the_same_samples_at_every_clock(self, lib):
        """Why a clock change is, for voiced sound, only a change of playback
        rate: the formant filters' corners and the sample rate both scale with
        the clock, so the coefficients do not move."""
        voiced = [0x24, 0x18, 0x35, 0x0C]          # AH L O1 M
        at_720 = self._stream(lib, 720000, voiced)
        assert np.array_equal(self._stream(lib, 360000, voiced), at_720)
        assert np.array_equal(self._stream(lib, 1440000, voiced), at_720)

    def test_fricatives_are_the_exception(self, lib):
        """The noise shaper has the clock in a numerator (MAME's reading of
        the die, votrax_filters.c), so S changes colour with the clock."""
        fricative = [0x1F, 0x24]                   # S AH
        assert not np.array_equal(self._stream(lib, 360000, fricative),
                                  self._stream(lib, 720000, fricative))

    def test_a_live_change_keeps_the_chip_talking(self, lib, chip):
        """Figures 6 and 7 vary the clock with a knob or a DAC while the chip
        speaks. The phone, its place and the queue carry on."""
        phones = (ctypes.c_ubyte * 3)(0x24, 0x24, 0x24)
        lib.vx_speak(chip, phones, 3)
        render(lib, chip, 6000)
        pending = lib.vx_pending(chip)
        lib.vx_set_clock(chip, 900000)
        after = render(lib, chip, 1000)
        assert lib.vx_pending(chip) == pending
        assert np.abs(after).max() > 1000, "the chip went quiet at the change"

        reference = lib.vx_create(0, 0)
        lib.vx_speak(reference, phones, 3)
        render(lib, reference, 6000)
        same = render(lib, reference, 1000)
        lib.vx_destroy(reference)
        assert np.array_equal(after, same)


# ------------------------------------------------------ Figure 8 output ---

class TestFigure8OutputStage:
    def _speech(self, lib, stage, clock=0):
        c = lib.vx_create(1, clock)
        lib.vx_set_output(c, stage)
        buf = (ctypes.c_ubyte * 4096)()
        n = lib.ttv_translate(b"She sells sea shells by the sea shore.", buf, 4096)
        lib.vx_speak(c, buf, n)
        out = []
        while lib.vx_pending(c):
            out.append(render(lib, c, 4096))
        rate = lib.vx_sample_rate(c)
        lib.vx_destroy(c)
        return np.concatenate(out), rate

    @staticmethod
    def band_energy(x, rate, lo, hi):
        spectrum = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / rate)
        return spectrum[(freqs >= lo) & (freqs < hi)].sum()

    def test_default_is_the_chip(self, lib, chip):
        assert lib.vx_output(chip) == 0
        lib.vx_set_output(chip, 1)
        assert lib.vx_output(chip) == 1
        lib.vx_set_output(chip, 7)
        assert lib.vx_output(chip) == 0

    def test_corners_derived_from_the_parts(self):
        """The numbers votrax.h documents, from the parts on page 10."""
        hp_in = 1 / (2 * math.pi * 1e-6 * (4700 + 1 / (1 / 10000 + 1 / 1200)))
        lp = 1 / (2 * math.pi * 0.1e-6 / (1 / 4700 + 1 / 10000 + 1 / 1200))
        hp_spk = 1 / (2 * math.pi * 330e-6 * 8)
        assert hp_in == pytest.approx(27.6, abs=0.05)
        assert lp == pytest.approx(1825, abs=1)
        assert hp_spk == pytest.approx(60.3, abs=0.05)

    def test_it_is_a_low_pass_around_two_kilohertz(self, lib):
        chip_out, rate = self._speech(lib, 0)
        fig8, _ = self._speech(lib, 1)
        mid = self.band_energy(fig8, rate, 300, 1000) / self.band_energy(chip_out, rate, 300, 1000)
        high = self.band_energy(fig8, rate, 6000, 12000) / self.band_energy(chip_out, rate, 6000, 12000)
        assert 0.5 < mid <= 1.05
        assert high < 0.1

    def test_the_speaker_coupling_removes_the_bias(self, lib):
        fig8, rate = self._speech(lib, 1)
        low = self.band_energy(fig8, rate, 0, 20)
        total = self.band_energy(fig8, rate, 0, rate / 2)
        assert low / total < 0.01

    def test_the_corners_do_not_move_with_the_clock(self, lib):
        """The loudspeaker circuit is fixed in hertz; the voice is not."""
        chip_hi, rate_hi = self._speech(lib, 0, 1440000)
        fig8_hi, _ = self._speech(lib, 1, 1440000)
        high = (self.band_energy(fig8_hi, rate_hi, 6000, 12000) /
                self.band_energy(chip_hi, rate_hi, 6000, 12000))
        assert high < 0.1

    def test_cancel_clears_it(self, lib, chip):
        lib.vx_set_output(chip, 1)
        lib.vx_speak(chip, (ctypes.c_ubyte * 2)(0x24, 0x24), 2)
        render(lib, chip, 4000)
        lib.vx_cancel(chip)
        assert np.abs(render(lib, chip, 400)).max() < 200
