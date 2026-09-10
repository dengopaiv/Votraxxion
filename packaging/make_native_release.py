"""Package the native artifacts into one versioned zip.

Run from the repo root, after building everything it collects:

    python nvda-addon/package.py          # both DLLs and the .nvda-addon
    gui-native\\build.cmd x64
    gui-native\\build.cmd x86
    python packaging/make_native_release.py

The zip lands at dist/votrax-sc01-<version>-win.zip.

This is the *native* release -- the C synthesizer and the things built
directly on it. It is deliberately separate from make_release.py, which
packages the wxPython Workbench: that one is a ~150 MB PyInstaller bundle
of Python, numpy, scipy and wx, and this one is under a megabyte because
nothing in it needs an interpreter. Two different artifacts with two
different audiences, so two different zips and two different versions.

Every file is listed with its SHA-256 in the release notes. The build is
unsigned, so a checksum somebody can actually check is the only thing
distinguishing this download from any other unsigned exe.
"""

from __future__ import annotations

import hashlib
import sys
import time
import zipfile
from pathlib import Path

# Matches nvda-addon/manifest.ini and the VERSIONINFO in
# gui-native/votrax_gui.rc. All three move together.
VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
OUT = DIST / f"votrax-sc01-{VERSION}-win.zip"

#: (source, name inside the zip). Missing files are a hard error rather
#: than a quiet gap -- a release zip with the x86 build silently absent is
#: worse than no zip.
CONTENTS = [
    (ROOT / "gui-native" / "build" / "votrax_gui-x64.exe",
     "gui/votrax_gui-x64.exe"),
    (ROOT / "gui-native" / "build" / "votrax_gui-x86.exe",
     "gui/votrax_gui-x86.exe"),
    (ROOT / "nvda-addon" / f"votraxsc01-{VERSION}.nvda-addon",
     f"nvda-addon/votraxsc01-{VERSION}.nvda-addon"),
    (ROOT / "nvda-addon" / "addon" / "synthDrivers" / "votraxsc01-x64.dll",
     "library/votraxsc01-x64.dll"),
    (ROOT / "nvda-addon" / "addon" / "synthDrivers" / "votraxsc01-x86.dll",
     "library/votraxsc01-x86.dll"),
    (ROOT / "src" / "votrax.h", "library/votrax.h"),
    (ROOT / "src" / "ttv.h", "library/ttv.h"),
    (ROOT / "LICENSE", "LICENSE"),
]

NOTES_HEAD = f"""\
Votrax SC-01 — native release {VERSION} (Windows)
=================================================

Build date: {time.strftime('%Y-%m-%d')}

An emulation of the Votrax SC-01, the 1980 phoneme chip behind the Type 'N
Talk, the Apple II Mockingboard's speech and a generation of arcade machines.
The speech is not imitated: the DSP is built from the schematics extracted
from die photographs and tracks MAME's silicon-level simulation of the
decapped part. Both production mask ROMs are included as voices.

Nothing here needs Python, a runtime, or a data file. Both mask ROMs and the
whole English letter-to-sound front end are compiled into the C, which is why
the whole release is under a megabyte.

What is in it
-------------

gui/            A standalone speech workbench. Type text or phonemes, pick a
                voice, hear it, save a WAV. One executable, no installer.
                Take votrax_gui-x64.exe unless you are on 32-bit Windows.

nvda-addon/     The NVDA screen-reader driver. Open the .nvda-addon file
                with NVDA running, or use Tools > Add-on store > Install
                from external source. Both mask revisions appear as voices.
                NVDA 2023.1 or later; the x86 library ships alongside for
                NVDA 2025 and earlier, which ran 32-bit.

library/        The synthesizer as a DLL, with its two headers, for anyone
                embedding it. The C API is documented in votrax.h; the
                entry points are vx_* for the chip and ttv_* for the text
                front end. No ROM file to find and nothing to initialise.

Running the GUI
---------------

Double-click gui\\votrax_gui-x64.exe. There is nothing to install and nothing
to extract beside it.

The build is unsigned, so Windows SmartScreen will warn the first time it
runs: click "More info" then "Run anyway" if you trust the source. The
checksums below are there so that trust can be checked rather than assumed.

Controls worth knowing
----------------------

Clock and Speed both make it faster and they are not the same thing. Moving
the master clock is what the 1980 hardware's single knob did: tempo and pitch
rise together and the voice turns into a chipmunk. Speed instead holds each
phoneme for less time and leaves the clock alone, so the tempo moves and the
pitch does not. Both are here as separate controls because they are separate
things.

Phoneme mode takes the datasheet's own names — H AH1 L OO PA1 — with an
optional :0 to :3 after a name for that phoneme's pitch. "Convert to
Phonemes" turns whatever text is in the box into exactly that notation, so it
can be edited by hand and spoken again.

Everything is reachable from the keyboard: every control has an Alt
accelerator and a label the screen reader announces with it.

Accessibility
-------------

This is a front end for a screen-reader voice, so the GUI is plain Win32
controls throughout and the tab order is checked by a tool that launches the
real window and posts real keypresses into it, rather than by inspection.

Known behaviour, not faults
---------------------------

The stop consonants — P, T, K, B, D, G — are silent when spoken on their own.
A stop is a closure: the chip gates the vocal tract off, and what a listener
hears as the burst is the *next* sound's onset leaving that closure. A stop
with nothing after it is silence on the real chip too.

Licence
-------

BSD-3-Clause. The die analysis, the ROM transcription and the filter
topologies are Olivier Galibert's work for MAME, and the attribution travels
with anything derived from them. See LICENSE.

Checksums (SHA-256)
-------------------

"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    missing = [src for src, _ in CONTENTS if not src.is_file()]
    if missing:
        print("Cannot package -- these have not been built:")
        for path in missing:
            print("  %s" % path.relative_to(ROOT))
        print("\nBuild them with:")
        print("  python nvda-addon/package.py")
        print("  gui-native\\build.cmd x64")
        print("  gui-native\\build.cmd x86")
        return 1

    lines = []
    for src, name in CONTENTS:
        lines.append("%s  %s (%s bytes)"
                     % (sha256(src), name, f"{src.stat().st_size:,}"))
    notes = NOTES_HEAD + "\n".join(lines) + "\n"

    DIST.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("RELEASE-NOTES.txt", notes.replace("\n", "\r\n"))
        for src, name in CONTENTS:
            z.write(src, name)

    print("wrote %s" % OUT.relative_to(ROOT))
    print("  %s bytes" % f"{OUT.stat().st_size:,}")
    for src, name in CONTENTS:
        print("  %-46s %9s bytes" % (name, f"{src.stat().st_size:,}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
