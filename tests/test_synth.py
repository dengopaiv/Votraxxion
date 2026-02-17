"""End-to-end tests for the VotraxSynthesizer."""

import os
import tempfile

import numpy as np
import pytest
from pyvotrax.synth import VotraxSynthesizer
from pyvotrax.phonemes import name_to_code, code_to_name, PHONE_TABLE


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
