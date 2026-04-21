"""Tests for the enhanced-DSP path (py_emu backend) wiring."""

import numpy as np
import pytest

from pyvotrax.synth import VotraxSynthesizer
from pyvotrax.tts import VotraxTTS
from pyvotrax.phonemes import name_to_code
from pyvotrax.presets import Preset, load_preset, factory_presets_dir


class TestEnhancedDspPath:
    def test_default_is_cpp_backend(self):
        synth = VotraxSynthesizer()
        assert synth.enhanced_dsp is False
        # C++ chip exposes master_clock; py_emu does not
        assert hasattr(synth._chip, "_native")

    def test_enhanced_dsp_uses_py_emu_backend(self):
        synth = VotraxSynthesizer(enhanced_dsp=True)
        assert synth.enhanced_dsp is True
        # py_emu chip has no _native attr
        assert not hasattr(synth._chip, "_native")
        # py_emu SCLOCK is 40 kHz
        assert synth.sclock == pytest.approx(40_000.0)

    def test_enhanced_dsp_produces_finite_nonsilent_audio(self):
        synth = VotraxSynthesizer(enhanced_dsp=True)
        audio = synth.synthesize_by_name(["AH", "STOP"])
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))
        rms = float(np.sqrt(np.mean(audio ** 2)))
        assert rms > 1e-6

    def test_enhanced_dsp_no_mid_utterance_clicks(self):
        """Regression: enhanced_dsp must not click at phoneme boundaries.

        Earlier, py_emu.phone_commit recomputed the LF glottal's f0/tp/alpha/E0
        from the SC-01A pitch formula (which depends on filt_f1 per phoneme),
        so every phoneme boundary discontinuously jumped the glottal waveform
        shape while _lf_phase kept rolling — audible as a click in the middle
        of breathy renders. The fix pins F0 to a constant across the utterance.
        """
        from pyvotrax.phonemes import parse_phoneme_sequence

        seq = "AH:2 E:2 I:2 O:2 U:2 STOP"  # breathy_lf preset's input
        synth = VotraxSynthesizer(enhanced_dsp=True, rd=2.0, dc_block=True)
        audio = synth.synthesize_phoneme_string(seq)
        # Compute RMS over 50-sample windows and check no window spikes to
        # >3x both neighbors (would be a click). Skip the very end (fade region).
        win = 50
        n_win = len(audio) // win
        rms = np.array([float(np.sqrt(np.mean(audio[i*win:(i+1)*win] ** 2)))
                        for i in range(n_win)])
        # Exclude the last 10 windows (fade) and first 2 (startup).
        body = rms[2:-10]
        for i in range(1, len(body) - 1):
            if body[i] > 0.002:  # ignore silence spikes
                assert body[i] < 3.0 * body[i - 1] or body[i] < 3.0 * body[i + 1], (
                    f"RMS spike at window {i + 2}: {body[i]:.4f} vs neighbors "
                    f"{body[i-1]:.4f} / {body[i+1]:.4f}"
                )

    def test_enhanced_vs_default_output_differs(self):
        """The LF glottal + oversampling + jitter should audibly diverge from
        the default 9-level stepped glottal."""
        ah = name_to_code("AH")
        default = VotraxSynthesizer().synthesize([(ah, 0)])
        enhanced = VotraxSynthesizer(enhanced_dsp=True).synthesize([(ah, 0)])
        # They should differ in length or waveform — trim to common length
        n = min(len(default), len(enhanced))
        diff = float(np.sqrt(np.mean((default[:n] - enhanced[:n]) ** 2)))
        assert diff > 1e-3

    def test_rd_is_forwarded(self):
        synth = VotraxSynthesizer(enhanced_dsp=True, rd=2.0)
        assert synth.rd == 2.0
        # py_emu stores rd as self._rd
        assert getattr(synth._chip, "_rd", None) == 2.0

    def test_override_in_enhanced_raises(self):
        synth = VotraxSynthesizer(enhanced_dsp=True)
        with pytest.raises(NotImplementedError):
            synth.synthesize([(name_to_code("AH"), 0, {"f1": 0})])

    def test_tts_forwards_enhanced_dsp(self):
        tts = VotraxTTS(enhanced_dsp=True, rd=1.5)
        assert tts._synth.enhanced_dsp is True
        assert tts._synth.rd == 1.5
        audio = tts.speak("Hi.")
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))

    def test_write_wav_uses_py_emu_sclock(self, tmp_path):
        synth = VotraxSynthesizer(enhanced_dsp=True)
        audio = synth.synthesize([(name_to_code("AH"), 0)])
        out = tmp_path / "enhanced.wav"
        synth.write_wav(audio, str(out), target_rate=44100)
        from scipy.io import wavfile
        rate, data = wavfile.read(str(out))
        assert rate == 44100
        assert len(data) > 0


class TestEnhancedDspPresets:
    def test_natural_lf_preset(self):
        p = load_preset(factory_presets_dir() / "natural_lf.json")
        assert p.synth.get("enhanced_dsp") is True
        synth = p.build_synthesizer()
        assert synth.enhanced_dsp is True

    def test_breathy_lf_preset_rd(self):
        p = load_preset(factory_presets_dir() / "breathy_lf.json")
        assert p.synth.get("rd") == 2.0
        synth = p.build_synthesizer()
        assert synth.rd == 2.0
