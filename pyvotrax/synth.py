"""High-level speech synthesizer using the Votrax SC-01A emulator."""

import numpy as np
from scipy.signal import resample_poly
from scipy.io import wavfile

from .chip import VotraxSC01A
from .phonemes import name_to_code
from .filters import SCLOCK


class VotraxSynthesizer:
    """High-level interface for synthesizing speech from phoneme sequences."""

    def __init__(self, enhanced: bool = False):
        self._chip = VotraxSC01A(enhanced=enhanced)

    def synthesize(self, phonemes: list) -> np.ndarray:
        """Synthesize audio from a list of (phoneme_code, inflection) tuples.

        Args:
            phonemes: List of (code, inflection) tuples where code is 0-63
                      and inflection is 0-3.

        Returns:
            Audio samples at 40 kHz as a float64 numpy array.
        """
        self._chip.reset()
        segments = []

        for phone, inflection in phonemes:
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
        tail_len = int(0.2 * SCLOCK)
        tail = np.empty(tail_len)
        for i in range(tail_len):
            tail[i] = self._chip.generate_one_sample()
        segments.append(tail)

        return np.concatenate(segments)

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

    @staticmethod
    def to_wav(audio: np.ndarray, filename: str, target_rate: int = 44100):
        """Write audio to a WAV file, resampling to the target rate.

        Args:
            audio: Audio samples at 40 kHz (float64)
            filename: Output WAV file path
            target_rate: Target sample rate (default 44100)
        """
        if len(audio) == 0:
            wavfile.write(filename, target_rate, np.array([], dtype=np.int16))
            return

        # Resample from SCLOCK (40000) to target_rate using rational resampling
        # Find GCD for rational resampling ratio
        from math import gcd
        g = gcd(target_rate, SCLOCK)
        up = target_rate // g
        down = SCLOCK // g

        resampled = resample_poly(audio, up, down)

        # RMS-based normalization: keeps voice and noise at natural relative levels
        # Peak normalization crushed vowels when noise spikes dominated
        rms = np.sqrt(np.mean(resampled ** 2))
        if rms > 0:
            target_rms = 32767 * 0.25  # -12dB below full scale
            resampled = resampled * (target_rms / rms)
            resampled = np.clip(resampled, -32767, 32767)

        wavfile.write(filename, target_rate, resampled.astype(np.int16))
