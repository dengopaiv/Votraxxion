"""High-level speech synthesizer using the Votrax SC-01A emulator."""

import numpy as np
from scipy.signal import resample_poly
from scipy.io import wavfile

from .chip import VotraxSC01A
from .phonemes import name_to_code
from .filters import SCLOCK


class VotraxSynthesizer:
    """High-level interface for synthesizing speech from phoneme sequences."""

    def __init__(self):
        self._chip = VotraxSC01A()

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
            n_samples = self._chip.get_phone_duration_samples()
            if n_samples > 0:
                segment = self._chip.generate_samples(n_samples)
                segments.append(segment)

        if not segments:
            return np.array([], dtype=np.float64)

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

        # Normalize to int16 range
        peak = np.max(np.abs(resampled))
        if peak > 0:
            resampled = resampled / peak * 32767.0

        wavfile.write(filename, target_rate, resampled.astype(np.int16))
