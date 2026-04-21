"""High-level speech synthesizer using the Votrax SC-01A emulator."""

import math
import numpy as np
from scipy.signal import resample_poly
from scipy.io import wavfile

from .chip import VotraxSC01A
from .phonemes import name_to_code, parse_phoneme_sequence
from .constants import MASTER_CLOCK as _DEFAULT_MASTER_CLOCK
from .constants import SCLOCK


class VotraxSynthesizer:
    """High-level interface for synthesizing speech from phoneme sequences.

    Args:
        dc_block: Apply a 20 Hz DC blocking filter to output. Default True.
            The 1980 SC-01A datasheet documents AO as a DC-biased pin; a real
            speaker amp removes the bias via its AC coupling, and we mimic that
            here so rendered audio doesn't click at start/stop from a ~1.5% DC
            step. Set False to preserve the raw DC-biased chip output.
        radiation_filter: Apply a +6 dB/oct radiation filter to output (default False).
        master_clock: Master clock frequency in Hz. Default 720 000. Varying it
            is the datasheet-endorsed way to produce sound-effect voices
            (slower clock → lower pitch + longer phonemes).
            Ignored when ``enhanced_dsp=True``.
        fx_fudge: Final-stage lowpass scaling. Default 150/4000 (authentic);
            1.0 gives the muffled "as-schematic" 150 Hz behavior.
            Ignored when ``enhanced_dsp=True``.
        closure_strength: Scales plosive closure attenuation. Default 1.0.
            Ignored when ``enhanced_dsp=True``.
        enhanced_dsp: Enable the pure-Python enhanced DSP path: Liljencrants-
            Fant glottal model (voice-quality controllable via ``rd``), PolyBLEP
            anti-aliasing, 2× oversampling, F0 jitter (~1.5%), amplitude shimmer
            (~2%), nasal anti-resonators (M/N/NG), per-fricative noise shapers
            (S/Z/SH/ZH/F/V/TH/THV). **Much slower** than the C++ path —
            acceptable for offline music-sample rendering, not for realtime NVDA
            playback. Also diverges from MAME-authentic sound. Default False.
        rd: Liljencrants-Fant voice-quality parameter when ``enhanced_dsp=True``.
            Range roughly 0.3 (pressed/tense) to 2.7 (breathy/lax); 1.0 is modal.
            Ignored when ``enhanced_dsp=False``.
    """

    def __init__(
        self,
        dc_block: bool = True,
        radiation_filter: bool = False,
        master_clock: float = _DEFAULT_MASTER_CLOCK,
        fx_fudge: float = 150.0 / 4000.0,
        closure_strength: float = 1.0,
        enhanced_dsp: bool = False,
        rd: float = 1.0,
    ):
        self._enhanced_dsp = bool(enhanced_dsp)
        self._rd = float(rd)
        if self._enhanced_dsp:
            # The py_emu path runs at its own fixed SCLOCK (40 kHz) and does not
            # support master_clock / fx_fudge / closure_strength / overrides.
            from py_emu import VotraxSC01APython
            from py_emu.filters import SCLOCK as _PY_SCLOCK
            self._chip = VotraxSC01APython(enhanced=True, rd=self._rd)
            self._sclock = float(_PY_SCLOCK)
        else:
            self._chip = VotraxSC01A(
                master_clock=master_clock,
                fx_fudge=fx_fudge,
                closure_strength=closure_strength,
            )
            self._sclock = self._chip.sclock
        self._dc_block = dc_block
        self._radiation_filter = radiation_filter

    @property
    def enhanced_dsp(self) -> bool:
        return self._enhanced_dsp

    @property
    def rd(self) -> float:
        return self._rd

    @property
    def sclock(self) -> float:
        """Effective analog sample rate of the chip's output stream."""
        return self._sclock

    @property
    def master_clock(self) -> float:
        """Effective master clock in Hz. Nominal 720 000 in the enhanced-DSP path."""
        return getattr(self._chip, "master_clock", self._sclock * 18.0)

    def synthesize(self, phonemes: list) -> np.ndarray:
        """Synthesize audio from a list of phoneme specifications.

        Each element may be:

        * ``(code, inflection)`` — standard form, looks up ROM parameters.
        * ``(code, inflection, overrides)`` — ``overrides`` is a dict of
          PhonemeParams fields to replace for this one phoneme (e.g.
          ``{"f1": 10, "duration": 80}``). Passing an empty or ``None`` dict
          is equivalent to the 2-tuple form.

        Args:
            phonemes: Iterable of phoneme specs as above.

        Returns:
            Audio samples at :attr:`sclock` as a float64 numpy array.
        """
        self._chip.reset()
        segments = []

        for spec in phonemes:
            if len(spec) == 2:
                phone, inflection = spec
                overrides = None
            elif len(spec) == 3:
                phone, inflection, overrides = spec
            else:
                raise ValueError(
                    f"Phoneme spec must be a 2- or 3-tuple, got {spec!r}"
                )

            if overrides:
                if self._enhanced_dsp:
                    raise NotImplementedError(
                        "Per-phoneme overrides are not supported in the "
                        "enhanced-DSP path. Build the synthesizer with "
                        "enhanced_dsp=False to use overrides."
                    )
                self._chip.phone_commit_override(phone, inflection, **overrides)
            else:
                self._chip.phone_commit(phone, inflection)

            samples = []
            while not self._chip.phone_done:
                samples.append(self._chip.generate_one_sample())
            if samples:
                segments.append(np.array(samples))

        if not segments:
            return np.array([], dtype=np.float64)

        # After all phonemes finish, the chip's filters still contain energy
        # that needs to decay naturally (just like real hardware runs continuously).
        # Generate 200ms of tail to capture this decay.
        tail_len = int(0.2 * self._sclock)
        tail = np.empty(tail_len)
        for i in range(tail_len):
            tail[i] = self._chip.generate_one_sample()
        segments.append(tail)

        audio = np.concatenate(segments)

        # Optional post-processing (not part of the chip, but useful for output)
        if self._radiation_filter:
            audio = self._apply_radiation(audio)
        if self._dc_block:
            audio = self._apply_dc_block(audio, self._sclock)

        # Short end fade-out. Even with dc_block, a low-level filter resonance
        # can persist past our 200 ms tail and click when playback stops. 5 ms
        # of raised-cosine taper at the end is inaudible as a fade but guarantees
        # a clean silence transition for both live playback and WAV export.
        audio = self._apply_end_fade(audio, self._sclock)

        return audio

    @staticmethod
    def _apply_radiation(audio: np.ndarray) -> np.ndarray:
        """First-difference radiation filter (+6 dB/oct)."""
        out = np.empty_like(audio)
        out[0] = audio[0]
        out[1:] = audio[1:] - audio[:-1]
        return out

    @staticmethod
    def _apply_end_fade(audio: np.ndarray, sclock: float, fade_ms: float = 5.0) -> np.ndarray:
        """Raised-cosine taper of the last ``fade_ms`` milliseconds to zero."""
        n = int(fade_ms * sclock / 1000.0)
        if n <= 1 or n >= len(audio):
            return audio
        window = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, n)))
        audio = audio.copy()
        audio[-n:] *= window
        return audio

    @staticmethod
    def _apply_dc_block(audio: np.ndarray, sclock: float = SCLOCK) -> np.ndarray:
        """DC blocking filter with 20 Hz cutoff."""
        R = 1.0 - (2.0 * math.pi * 20.0 / sclock)
        out = np.empty_like(audio)
        prev_in = 0.0
        prev_out = 0.0
        for i in range(len(audio)):
            out[i] = audio[i] - prev_in + R * prev_out
            prev_in = audio[i]
            prev_out = out[i]
        return out

    def synthesize_by_name(self, names: list, inflection: int = 0) -> np.ndarray:
        """Synthesize audio from a list of phoneme names.

        Args:
            names: List of phoneme name strings (e.g., ["AH", "L", "STOP"])
            inflection: Default inflection for all phonemes (0-3)

        Returns:
            Audio samples at 40 kHz as a float64 numpy array.
        """
        phonemes = [(name_to_code(name), inflection) for name in names]
        return self.synthesize(phonemes)

    def synthesize_phoneme_string(self, seq: str) -> np.ndarray:
        """Synthesize audio directly from a Votrax phoneme-sequence string.

        Bypasses the CMU dict / TTS pipeline — the string is parsed straight
        into (code, inflection) tuples by
        :func:`pyvotrax.phonemes.parse_phoneme_sequence` and fed to the chip.
        Intended for music-production sample design where authorship happens at
        the phoneme level.

        See :func:`pyvotrax.phonemes.parse_phoneme_sequence` for token syntax.
        """
        return self.synthesize(parse_phoneme_sequence(seq))

    def write_wav(
        self,
        audio: np.ndarray,
        filename: str,
        target_rate: int = 44100,
    ):
        """Write audio to WAV, resampling from this synth's effective SCLOCK.

        Prefer this over :meth:`to_wav` when you constructed the synth with a
        non-default ``master_clock``.
        """
        self.to_wav(audio, filename, target_rate=target_rate, source_rate=self._sclock)

    @staticmethod
    def to_wav(
        audio: np.ndarray,
        filename: str,
        target_rate: int = 44100,
        source_rate: float = SCLOCK,
    ):
        """Write audio to a WAV file, resampling to the target rate.

        Args:
            audio: Audio samples at ``source_rate`` (float64).
            filename: Output WAV file path.
            target_rate: Target sample rate (default 44100).
            source_rate: Sample rate of ``audio`` in Hz. Defaults to the nominal
                40 kHz SCLOCK; if you constructed the chip with a non-default
                master clock, pass ``synth.sclock`` here.
        """
        if len(audio) == 0:
            wavfile.write(filename, target_rate, np.array([], dtype=np.int16))
            return

        # Rational resampling from source_rate → target_rate
        from math import gcd
        source_rate_int = int(round(source_rate))
        g = gcd(int(target_rate), source_rate_int)
        up = int(target_rate) // g
        down = source_rate_int // g

        resampled = resample_poly(audio, up, down)

        # RMS-based normalization: keeps voice and noise at natural relative levels
        # Peak normalization crushed vowels when noise spikes dominated
        rms = np.sqrt(np.mean(resampled ** 2))
        if rms > 0:
            target_rms = 32767 * 0.25  # -12dB below full scale
            resampled = resampled * (target_rms / rms)
            resampled = np.clip(resampled, -32767, 32767)

        wavfile.write(filename, target_rate, resampled.astype(np.int16))
