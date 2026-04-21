"""JSON preset save/load for the votraxxion music-production workbench.

A preset captures the full user-addressable parameter surface plus the input
source (text or phoneme string). Presets are JSON files with a versioned
schema; factory presets ship in-tree under ``presets/factory/``, user presets
live under ``%APPDATA%/votraxxion/presets/`` on Windows (or
``$XDG_DATA_HOME/votraxxion/presets`` / ``~/.local/share/votraxxion/presets``
elsewhere).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .synth import VotraxSynthesizer
from .tts import VotraxTTS

PRESET_SCHEMA_VERSION = 1

# Keys accepted inside the "synth" block. Extra keys are forwarded blindly
# so new chip parameters (added in later tasks) don't require a schema bump,
# but known keys are coerced to the right type up front.
_SYNTH_TYPED_KEYS: dict[str, type] = {
    "master_clock": float,
    "fx_fudge": float,
    "closure_strength": float,
    "dc_block": bool,
    "radiation_filter": bool,
    "enhanced_dsp": bool,
    "rd": float,
}

_TTS_TYPED_KEYS: dict[str, type] = {
    "enhanced": bool,
}


@dataclass
class Preset:
    """Serializable parameter bundle for the music-production workbench."""

    name: str = "Untitled"
    description: str = ""
    mode: str = "phoneme_string"  # "phoneme_string" or "text"
    input: str = ""
    synth: dict[str, Any] = field(default_factory=dict)
    tts: dict[str, Any] = field(default_factory=dict)
    schema_version: int = PRESET_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Preset":
        version = data.get("schema_version", PRESET_SCHEMA_VERSION)
        if version != PRESET_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported preset schema_version {version}; expected "
                f"{PRESET_SCHEMA_VERSION}"
            )
        mode = data.get("mode", "phoneme_string")
        if mode not in ("phoneme_string", "text"):
            raise ValueError(f"Invalid preset mode: {mode!r}")
        synth = _coerce_kwargs(data.get("synth") or {}, _SYNTH_TYPED_KEYS)
        tts = _coerce_kwargs(data.get("tts") or {}, _TTS_TYPED_KEYS)
        return cls(
            name=str(data.get("name", "Untitled")),
            description=str(data.get("description", "")),
            mode=mode,
            input=str(data.get("input", "")),
            synth=synth,
            tts=tts,
            schema_version=version,
        )

    def build_synthesizer(self) -> VotraxSynthesizer:
        """Construct a VotraxSynthesizer from this preset's synth parameters."""
        return VotraxSynthesizer(**self.synth)

    def build_tts(self) -> VotraxTTS:
        """Construct a VotraxTTS combining tts-level and synth-level params."""
        return VotraxTTS(**{**self.tts, **self.synth})


def _coerce_kwargs(d: dict[str, Any], typed: dict[str, type]) -> dict[str, Any]:
    """Coerce known keys to expected types; pass unknown keys through unchanged."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k in typed:
            t = typed[k]
            out[k] = t(v) if v is not None else None
        else:
            out[k] = v
    return out


def load_preset(path: str | os.PathLike) -> Preset:
    """Load a preset from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Preset.from_dict(data)


def save_preset(path: str | os.PathLike, preset: Preset) -> None:
    """Write a preset to a JSON file (pretty-printed, UTF-8)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preset.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def preset_filename(name: str) -> str:
    """Convert a preset's display name into a safe filename (no extension)."""
    slug = _SAFE_NAME_RE.sub("_", name.strip()).strip("_.")
    return slug or "untitled"


def user_presets_dir() -> Path:
    """Return the per-user presets directory, creating it if missing.

    Windows: ``%APPDATA%/votraxxion/presets``
    macOS / Linux: ``$XDG_DATA_HOME/votraxxion/presets`` or
                   ``~/.local/share/votraxxion/presets``.
    """
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData/Roaming")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share")
    d = Path(base) / "votraxxion" / "presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def factory_presets_dir() -> Path:
    """Return the in-tree factory-presets directory."""
    return Path(__file__).resolve().parent.parent / "presets" / "factory"


def list_presets(directory: Path | None = None) -> list[Path]:
    """List preset JSON files in ``directory`` (defaults to user dir + factory dir).

    Order: user presets first (so user names shadow factory names if duplicated),
    then factory presets. Files are sorted by stem name within each source.
    """
    if directory is not None:
        return sorted(Path(directory).glob("*.json"))
    out: list[Path] = []
    user_dir = user_presets_dir()
    factory_dir = factory_presets_dir()
    out.extend(sorted(user_dir.glob("*.json")))
    if factory_dir.exists():
        out.extend(sorted(factory_dir.glob("*.json")))
    return out
