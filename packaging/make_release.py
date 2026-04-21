"""Package dist/VotraxWorkbench/ into a versioned Windows zip.

Run from the repo root *after* the PyInstaller build:

    python -m PyInstaller packaging/votrax-gui.spec
    python packaging/make_release.py

The zip lands at dist/VotraxWorkbench-<version>-win64.zip. The folder name
inside the zip is VotraxWorkbench/, so an extract-and-run workflow leaves
the user with dist-like layout without a nested directory.
"""

from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path


# --- Configuration -----------------------------------------------------------

# Zip-level version. Independent of pyvotrax's pyproject.toml version because
# the standalone workbench .exe is a distinct artifact from the pyvotrax
# Python package / NVDA addon.
VERSION = "0.3.0a1"
ARCH = "win64"

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "dist" / "VotraxWorkbench"
OUT_PATH = REPO_ROOT / "dist" / f"VotraxWorkbench-{VERSION}-{ARCH}.zip"

RELEASE_NOTES = f"""\
Votrax SC-01A Workbench — prerelease {VERSION} ({ARCH})
=======================================================

Build date: {time.strftime('%Y-%m-%d')}

This is a preview build, not a stable release. It packages the wxPython GUI,
the compiled C++ DSP core (pybind11), cmudict, numpy, scipy, sounddevice, and
the factory presets into a one-dir layout.

Running
-------

1. Extract this zip to any folder.
2. Double-click VotraxWorkbench\\VotraxWorkbench.exe.

The entire VotraxWorkbench\\ directory must stay together - the exe depends on
the _internal\\ folder next to it.

The installer is unsigned. Windows SmartScreen may warn on first launch; click
"More info" then "Run anyway" if you trust the source.

What is this
------------

Music-production sound-design workbench for the Votrax SC-01A speech
synthesizer chip (1980). Two input modes (English text via CMU dict, or
direct phoneme-string entry), a dense parameter panel exposing master clock /
FX cutoff fudge / closure strength / LF enhanced-DSP path / TTS prosody /
output gain, preset save/load, and WAV export at 22.05/44.1/48/96 kHz in
16-bit, 32-bit int, or 32-bit float.

Known limitations
-----------------

- Per-phoneme parameter overrides are exposed in the Python API but not yet
  in the GUI - planned for a future release.
- Sounddevice-side end-of-stream hardware clicks are mitigated with a 150 ms
  trailing silence pad; some Windows exclusive-mode audio drivers may still
  pop on stream teardown.
- Plosive onset clicks (D/B/G/T/K/P) are authentic chip-level artifacts.
  Soften via the Closure Strength slider or load the "Soft plosives" preset.

See README.md for build-from-source instructions.
"""


def human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def main() -> int:
    if not SRC_DIR.is_dir():
        print(
            f"ERROR: {SRC_DIR} does not exist.\n"
            f"       Run `python -m PyInstaller packaging/votrax-gui.spec` first.",
            file=sys.stderr,
        )
        return 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        OUT_PATH.unlink()

    print(f"Packaging {SRC_DIR} ...")
    total_bytes = 0
    file_count = 0
    t0 = time.time()

    with zipfile.ZipFile(
        OUT_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        # Release notes at the top of the zip.
        zf.writestr("RELEASE-NOTES.txt", RELEASE_NOTES)
        # Walk the bundle.
        for root, _, files in os.walk(SRC_DIR):
            for name in files:
                src = Path(root) / name
                rel = src.relative_to(SRC_DIR.parent)  # so top-level is VotraxWorkbench/
                zf.write(src, arcname=str(rel).replace(os.sep, "/"))
                total_bytes += src.stat().st_size
                file_count += 1

    elapsed = time.time() - t0
    zip_size = OUT_PATH.stat().st_size
    print(f"  files zipped     : {file_count}")
    print(f"  uncompressed size: {human_bytes(total_bytes)}")
    print(f"  compressed size  : {human_bytes(zip_size)}")
    print(f"  compression      : {100 * (1 - zip_size / total_bytes):.1f}%")
    print(f"  time             : {elapsed:.1f}s")
    print()
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
