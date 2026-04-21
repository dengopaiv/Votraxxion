"""Votrax SC-01A NVDA speech synthesizer driver.

This synth driver uses the pyvotrax chip emulator and CMU dictionary
to provide text-to-speech via the Votrax SC-01A DSP emulation.
"""

import os
import sys
import threading
import queue
from math import gcd

import numpy as np

# Add bundled library path so pyvotrax and deps can be imported
_addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_lib_dir = os.path.join(_addon_dir, "lib")
if os.path.isdir(_lib_dir) and _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)

import synthDriverHandler
from speech.commands import IndexCommand
import nvwave
from logHandler import log

from pyvotrax.tts import VotraxTTS
from pyvotrax.synth import VotraxSynthesizer
from pyvotrax.constants import SCLOCK


class SynthDriver(synthDriverHandler.SynthDriver):
    """Votrax SC-01A speech synthesizer for NVDA."""

    name = "votrax"
    description = "Votrax SC-01A"

    supportedSettings = [
        synthDriverHandler.SynthDriver.RateSetting(),
        synthDriverHandler.SynthDriver.PitchSetting(),
        synthDriverHandler.SynthDriver.VolumeSetting(),
        synthDriverHandler.BooleanSynthSetting("enhanced", "Enhanced mode", defaultVal=True),
    ]

    @classmethod
    def check(cls):
        """Check if the synth driver can be used."""
        try:
            from pyvotrax.tts import VotraxTTS
            return True
        except ImportError:
            return False

    def __init__(self):
        super().__init__()
        self._enhanced = True
        self._tts = VotraxTTS(enhanced=self._enhanced)
        self._rate = 50
        self._pitch = 50
        self._volume = 100
        self._lock = threading.Lock()

        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        self._player = nvwave.WavePlayer(
            channels=1,
            samplesPerSec=44100,
            bitsPerSample=16,
        )

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def terminate(self):
        self._stop_event.set()
        self._queue.put(None)  # Sentinel to unblock worker
        self._thread.join(timeout=2)
        self._player.close()

    def _process_audio(self, audio):
        """Apply pitch, rate, and volume to raw audio.

        Returns int16 bytes at 44100 Hz for WavePlayer.
        """
        from scipy.signal import resample_poly

        if len(audio) == 0:
            return b""

        # Pitch: resample from shifted source rate to 44100
        pitch_factor = 0.5 + (self._pitch / 100.0)
        source_rate = int(SCLOCK * pitch_factor)
        target_rate = 44100

        g = gcd(target_rate, source_rate)
        up = target_rate // g
        down = source_rate // g
        resampled = resample_poly(audio, up, down)

        # Rate: second resample to change duration
        speed_factor = 0.5 + (self._rate / 100.0)
        speed_down = int(1000 * speed_factor)
        if speed_down != 1000:
            resampled = resample_poly(resampled, 1000, speed_down)

        # Volume: normalize then scale
        peak = np.max(np.abs(resampled))
        if peak > 0:
            resampled = resampled / peak * 32767.0 * (self._volume / 100.0)

        return resampled.astype(np.int16).tobytes()

    def _worker(self):
        """Background worker that processes the speech queue."""
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is None:
                break

            text, index_callback = item

            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            try:
                audio = self._tts.speak(text)
                if len(audio) == 0:
                    continue

                samples = self._process_audio(audio)

                if not self._stop_event.is_set() and samples:
                    self._player.feed(samples)

                if index_callback is not None:
                    index_callback()

            except Exception:
                log.error("Votrax: synthesis error", exc_info=True)

    def speak(self, speechSequence):
        """Process an NVDA speech sequence."""
        text_parts = []
        pending_index = None

        for item in speechSequence:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, IndexCommand):
                # Speak accumulated text before index
                if text_parts:
                    full_text = " ".join(text_parts)
                    self._queue.put((full_text, None))
                    text_parts = []
                pending_index = item.index

        # Speak remaining text
        if text_parts:
            full_text = " ".join(text_parts)
            callback = None
            if pending_index is not None:
                idx = pending_index
                callback = lambda: synthDriverHandler.synthIndexReached(self, idx)
                pending_index = None
            self._queue.put((full_text, callback))

    def cancel(self):
        """Stop all speech."""
        # Clear the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._player.stop()

    def pause(self, switch):
        """Pause or resume speech."""
        if switch:
            self._pause_event.clear()
            self._player.pause(True)
        else:
            self._pause_event.set()
            self._player.pause(False)

    def _get_rate(self):
        return self._rate

    def _set_rate(self, value):
        self._rate = value

    def _get_pitch(self):
        return self._pitch

    def _set_pitch(self, value):
        self._pitch = value

    def _get_volume(self):
        return self._volume

    def _set_volume(self, value):
        self._volume = value

    def _get_enhanced(self):
        return self._enhanced

    def _set_enhanced(self, value):
        with self._lock:
            self._enhanced = value
            self._tts = VotraxTTS(enhanced=value)
