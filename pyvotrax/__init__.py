"""pyvotrax — Votrax SC-01A speech synthesizer DSP emulator.

Provides chip-level emulation of the Votrax SC-01A using a compiled C++
backend for high-performance sample generation.
"""

from .chip import VotraxSC01A
from .synth import VotraxSynthesizer
from .phonemes import PHONE_TABLE, name_to_code, code_to_name, parse_phoneme_sequence
from .tts import VotraxTTS
from .presets import (
    Preset,
    PRESET_SCHEMA_VERSION,
    load_preset,
    save_preset,
    list_presets,
    user_presets_dir,
    factory_presets_dir,
    preset_filename,
)


def has_native_core() -> bool:
    """Return True if the C++ DSP core extension is available.

    Always True — the C++ extension is now required.
    """
    return True


__all__ = [
    "VotraxSC01A",
    "VotraxSynthesizer",
    "VotraxTTS",
    "PHONE_TABLE",
    "name_to_code",
    "code_to_name",
    "parse_phoneme_sequence",
    "Preset",
    "PRESET_SCHEMA_VERSION",
    "load_preset",
    "save_preset",
    "list_presets",
    "user_presets_dir",
    "factory_presets_dir",
    "preset_filename",
    "has_native_core",
]
