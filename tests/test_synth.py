"""End-to-end tests for the VotraxSynthesizer."""

import os
import tempfile

import numpy as np
import pytest
from pyvotrax.synth import VotraxSynthesizer
from pyvotrax.phonemes import (
    name_to_code,
    code_to_name,
    parse_phoneme_sequence,
    PHONE_TABLE,
)


class TestPhonemes:
    def test_table_size(self):
        assert len(PHONE_TABLE) == 64

    def test_name_to_code_roundtrip(self):
        for i, name in enumerate(PHONE_TABLE):
            assert name_to_code(name) == i
            assert code_to_name(i) == name

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            name_to_code("NONEXISTENT")

    def test_out_of_range_code_raises(self):
        with pytest.raises(IndexError):
            code_to_name(64)
        with pytest.raises(IndexError):
            code_to_name(-1)


class TestParsePhonemeSequence:
    def test_single_bare_name(self):
        assert parse_phoneme_sequence("AH") == [(name_to_code("AH"), 0)]

    def test_inflection_suffix(self):
        assert parse_phoneme_sequence("AH:2") == [(name_to_code("AH"), 2)]

    def test_repeat_suffix(self):
        code = name_to_code("AH")
        assert parse_phoneme_sequence("AH*3") == [(code, 0)] * 3

    def test_inflection_and_repeat(self):
        code = name_to_code("AH")
        assert parse_phoneme_sequence("AH:2*3") == [(code, 2)] * 3

    def test_multiple_tokens(self):
        expected = [
            (name_to_code("I3"), 0),
            (name_to_code("M"), 0),
            (name_to_code("P"), 0),
            (name_to_code("O1"), 2),
            (name_to_code("R"), 1),
            (name_to_code("T"), 0),
        ]
        assert parse_phoneme_sequence("I3 M P O1:2 R:1 T") == expected

    def test_case_insensitive(self):
        assert parse_phoneme_sequence("ah:1") == [(name_to_code("AH"), 1)]

    def test_newlines_and_comments(self):
        src = """
        # word: TOP
        T O1:2 P
        # trailing stop
        STOP
        """
        result = parse_phoneme_sequence(src)
        assert result == [
            (name_to_code("T"), 0),
            (name_to_code("O1"), 2),
            (name_to_code("P"), 0),
            (name_to_code("STOP"), 0),
        ]

    def test_empty_string(self):
        assert parse_phoneme_sequence("") == []

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            parse_phoneme_sequence("NONEXISTENT")

    def test_malformed_token_raises(self):
        with pytest.raises(ValueError):
            parse_phoneme_sequence("AH:")

    def test_inflection_out_of_range_raises(self):
        with pytest.raises(ValueError):
            parse_phoneme_sequence("AH:4")


class TestSynthesize:
    def test_ah_vowel_not_silent(self):
        """Synthesize 'AH' vowel and verify non-silent output."""
        synth = VotraxSynthesizer()
        audio = synth.synthesize([(0x24, 0)])  # AH
        assert len(audio) > 0
        rms = np.sqrt(np.mean(audio ** 2))
        assert rms > 1e-10, f"AH vowel is silent (RMS={rms})"

    def test_synthesize_by_name(self):
        """Synthesize by name and verify output."""
        synth = VotraxSynthesizer()
        audio = synth.synthesize_by_name(["AH"])
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))

    def test_synthesize_sequence(self):
        """Synthesize a multi-phoneme sequence."""
        synth = VotraxSynthesizer()
        audio = synth.synthesize_by_name(["EH1", "L", "OO", "STOP"])
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))

    def test_empty_sequence(self):
        """Empty phoneme list should return empty array."""
        synth = VotraxSynthesizer()
        audio = synth.synthesize([])
        assert len(audio) == 0

    def test_synthesize_phoneme_string(self):
        """synthesize_phoneme_string should produce audio matching the parsed tuples."""
        synth = VotraxSynthesizer()
        audio_str = synth.synthesize_phoneme_string("AH:0 STOP")
        audio_tuples = synth.synthesize([
            (name_to_code("AH"), 0),
            (name_to_code("STOP"), 0),
        ])
        assert audio_str.shape == audio_tuples.shape
        assert np.allclose(audio_str, audio_tuples)

    def test_synthesize_phoneme_string_with_inflection(self):
        synth = VotraxSynthesizer()
        audio = synth.synthesize_phoneme_string("AH:2 STOP")
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))

    def test_dc_block_defaults_to_true(self):
        """Regression: dc_block must default to True. The SC-01A AO pin is
        DC-biased per the 1980 datasheet; not removing it clicks audibly at
        start/stop of playback and dirties WAV exports."""
        synth = VotraxSynthesizer()
        assert synth._dc_block is True
        audio = synth.synthesize_by_name(["AH", "STOP"])
        dc_mean = float(np.mean(audio))
        assert abs(dc_mean) < 1e-3, (
            f"Rendered audio has DC offset {dc_mean:+.5f}; dc_block did not run"
        )

    def test_dc_block_can_be_disabled(self):
        """The raw DC-biased chip output is still reachable with dc_block=False."""
        synth = VotraxSynthesizer(dc_block=False)
        audio = synth.synthesize_by_name(["AH", "STOP"])
        dc_mean = float(np.mean(audio))
        # With dc_block off the chip's native DC bias shows through;
        # confirm it really is larger than dc_block=True would give.
        assert abs(dc_mean) > 1e-3

    def test_end_fade_zeros_last_samples(self):
        """Regression: the last ~5 ms must taper to zero so playback doesn't
        click when the audio stream stops. Previously, filter resonance tails
        could leave non-zero end samples even after the 200 ms decay buffer."""
        synth = VotraxSynthesizer()
        audio = synth.synthesize_by_name(["AH", "STOP"])
        # The very last sample is the cosine endpoint — must be 0.
        assert audio[-1] == 0.0
        # The preceding few samples are in the near-zero end of the raised
        # cosine — inaudibly small (subnormal floats in practice).
        assert float(np.max(np.abs(audio[-5:]))) < 1e-6
        # Whole 5 ms fade region should be quiet.
        fade_samples = int(0.005 * synth.sclock)
        assert fade_samples > 10
        fade_region = audio[-fade_samples:]
        assert float(np.max(np.abs(fade_region))) < 0.01

    def test_master_clock_plumbing(self):
        """VotraxSynthesizer should propagate master_clock to the chip."""
        synth = VotraxSynthesizer(master_clock=360_000.0)
        assert synth.master_clock == 360_000.0
        assert synth.sclock == pytest.approx(20_000.0)

    def test_custom_clock_end_to_end(self):
        """Synthesize at half master clock — should produce finite, non-silent audio."""
        synth = VotraxSynthesizer(master_clock=360_000.0)
        audio = synth.synthesize_by_name(["AH", "STOP"])
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))
        rms = np.sqrt(np.mean(audio ** 2))
        assert rms > 1e-10

    def test_tail_extends_output(self):
        """Output should include ~200ms tail beyond the phoneme timing window."""
        from pyvotrax.filters import SCLOCK
        synth = VotraxSynthesizer()

        # Synthesize a single vowel followed by STOP
        audio = synth.synthesize_by_name(["AH", "STOP"])
        # The tail adds 0.2 * SCLOCK = 8000 samples
        tail_len = int(0.2 * SCLOCK)
        assert len(audio) >= tail_len, (
            f"Output length {len(audio)} is shorter than tail alone ({tail_len})"
        )


class TestToWav:
    def test_write_wav_file(self):
        """Synthesize and write a WAV file, verify it exists and has content."""
        synth = VotraxSynthesizer()
        audio = synth.synthesize([(0x24, 0)])  # AH

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmpfile = f.name

        try:
            synth.to_wav(audio, tmpfile)
            assert os.path.exists(tmpfile)
            assert os.path.getsize(tmpfile) > 44  # WAV header is 44 bytes
        finally:
            os.unlink(tmpfile)

    def test_write_empty_wav(self):
        """Writing an empty audio array should not crash."""
        synth = VotraxSynthesizer()
        audio = np.array([], dtype=np.float64)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmpfile = f.name

        try:
            synth.to_wav(audio, tmpfile)
            assert os.path.exists(tmpfile)
        finally:
            os.unlink(tmpfile)

    def test_write_wav_with_custom_clock(self):
        """write_wav instance method should resample from the synth's effective SCLOCK."""
        from scipy.io import wavfile

        synth = VotraxSynthesizer(master_clock=360_000.0)
        audio = synth.synthesize([(0x24, 0)])  # AH

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmpfile = f.name

        try:
            synth.write_wav(audio, tmpfile, target_rate=44100)
            rate, data = wavfile.read(tmpfile)
            assert rate == 44100
            # At half master clock, the same phoneme runs for the same number of
            # SCLOCK samples, but each SCLOCK sample represents 1/20000 s instead
            # of 1/40000 s. Resampled to 44100 Hz it will be ~2x longer than
            # the default-clock version. Just confirm it's non-trivially long.
            assert len(data) > 10_000

        finally:
            os.unlink(tmpfile)

    def test_wav_valid_format(self):
        """Written WAV should be readable by scipy."""
        from scipy.io import wavfile

        synth = VotraxSynthesizer()
        audio = synth.synthesize([(0x24, 0)])  # AH

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmpfile = f.name

        try:
            synth.to_wav(audio, tmpfile, target_rate=44100)
            rate, data = wavfile.read(tmpfile)
            assert rate == 44100
            assert len(data) > 0
            assert data.dtype == np.int16
        finally:
            os.unlink(tmpfile)
