# Votrax SC-01A Workbench

A Python + C++ emulator of the Votrax SC-01A speech synthesizer chip, packaged
as a Windows music-production workbench and an NVDA addon.

The DSP core is hand-built from the schematics extracted from die photographs,
published at [og.kervella.org/sc01a](http://og.kervella.org/sc01a). The C++
port tracks [MAME's votrax.cpp](https://github.com/mamedev/mame/blob/master/src/devices/sound/votrax.cpp).
For deeper background, see `docs/tech-overview.md`.

## Running from source

Prerequisites:

- Python 3.9+
- A C++ compiler toolchain for the pybind11 extension (on Windows: MSVC via
  Visual Studio Build Tools).

Install and build:

```
pip install -e .[gui,dev]
python setup.py build_ext --inplace
python -m pytest tests/
python -m pyvotrax
```

`pip install -e .[gui]` pulls in `wxPython` and `sounddevice`, which the GUI
needs; the core package without `[gui]` suffices for library / NVDA-addon use.

## Building the Windows app

The standalone `.exe` is produced in two stages: PyInstaller freezes the
wxPython GUI and bundles the compiled `_votrax_core.pyd`, cmudict, numpy,
scipy, sounddevice, and the factory presets into a one-dir layout; Inno Setup
then wraps that layout into a Windows installer.

### Stage 1 — PyInstaller one-dir bundle

Prerequisites:

- Everything from "Running from source" above.
- `pip install pyinstaller` (tested with 6.18).
- `python setup.py build_ext --inplace` has been run at least once so that
  `pyvotrax/_votrax_core.cp*-win_amd64.pyd` exists.

Build from the repo root:

```
python -m PyInstaller --clean --noconfirm packaging/votrax-gui.spec
```

Output: `dist/VotraxWorkbench/VotraxWorkbench.exe` plus a sibling `_internal/`
directory (~150 MB total). The .exe runs stand-alone; the whole folder must be
kept together.

Run it with a double-click, or from a shell:

```
dist/VotraxWorkbench/VotraxWorkbench.exe
```

### Stage 1b — prerelease zip (optional)

For a portable-zip release (no installer), after stage 1 run:

```
python packaging/make_release.py
```

Output: `dist/VotraxWorkbench-<version>-win64.zip`. Contents: a top-level
`RELEASE-NOTES.txt` and the whole `VotraxWorkbench/` folder. Extract anywhere
and double-click `VotraxWorkbench.exe`. The zip version is set inside
`packaging/make_release.py` and is independent of `pyproject.toml` because the
standalone workbench is a distinct artifact from the `pyvotrax` Python package
and the NVDA addon.

### Stage 2 — Inno Setup installer (optional)

Prerequisites:

- Inno Setup 6 installed — [jrsoftware.org/isinfo.php](https://jrsoftware.org/isinfo.php).

Build from the repo root after the PyInstaller stage has run:

```
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
```

Output: `dist/VotraxWorkbenchSetup-<version>.exe`. This is an unsigned installer;
Windows SmartScreen will prompt the first time it runs. Code signing is out of
scope for this build — add a signed step if you are distributing the installer
beyond your own machine.

## Building the standalone synthesizer

The synthesizer proper is C++ and has no dependencies — not on Python, not on
a phoneme dictionary, not on a ROM file. Both SC-01 mask ROMs and the whole
English front end are compiled in, so the result is one library and nothing
beside it. `csrc/votrax_capi.h` is the C API; `docs/tech-overview.md`, Part 4, has
the details.

Windows, MSVC (from a Developer Command Prompt):

```
cl /std:c++17 /EHsc /O2 /LD /Icsrc /Fe:votraxsc01.dll csrcotrax_capi.cpp
```

Linux or macOS:

```
c++ -std=c++17 -O2 -shared -fPIC -Icsrc -o libvotraxsc01.so csrc/votrax_capi.cpp
```

Driving it is the loop any Votrax front end has always used — hand the chip a
phone when it asks for one, and render audio in between:

```python
import ctypes
lib = ctypes.CDLL("./votraxsc01.dll")
lib.vx_create.restype = ctypes.c_void_p
lib.vx_create.argtypes = [ctypes.c_int, ctypes.c_uint]
lib.vx_sample_rate.restype = ctypes.c_double
lib.vx_sample_rate.argtypes = [ctypes.c_void_p]

chip = lib.vx_create(1, 0)          # 1 = the 1980 SC-01 mask, 0 = default clock
buf = ctypes.create_string_buffer(4096)
n = lib.ttv_translate(b"Hello.", buf, 4096)
```

Declare every `argtypes` before calling: without them ctypes guesses, and a
guessed 32-bit handle in a 64-bit process is a crash that only shows up once
the heap wanders past 4 GB.

## Building the NVDA addon

The native add-on is one Python file and one DLL per architecture — about
169 KB in total, with no bundled wheels, no pronunciation dictionary and no
ROM files:

```
cd nvda-addon
python package.py
```

That builds both libraries (MSVC required; the script finds vcvars itself) and
writes `votraxsc01-1.0.0.nvda-addon`. NVDA 2026 is 64-bit only, so the x64
library is the one it loads and the packager refuses to produce an add-on
without it; the x86 library ships alongside for NVDA 2025 and earlier, which
ran 32-bit. The driver picks between them from the bitness of the process it
finds itself in, so one add-on serves both.

The add-on offers both mask revisions as voices, rate as constant-pitch
truncation (with an "authentic rate" checkbox for the 1980 clock-scaling
behaviour), and pitch quantised to the chip's four real inflection levels.

`tests/test_nvda_driver.py` exercises the driver against stubbed NVDA modules,
so the shim can be tested without a screen reader; it skips if the DLL has not
been built.

### The older Python-based addon

`nvda-addon/` holds the previous driver, which used the pyvotrax emulator with
numpy, scipy and CMUdict bundled alongside (~40 MB). It is superseded by the
native add-on above but still builds with `python nvda-addon/package.py`.
