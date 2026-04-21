# Votrax SC-01A Workbench

A Python + C++ emulator of the Votrax SC-01A speech synthesizer chip, packaged
as a Windows music-production workbench and an NVDA addon.

The DSP core is hand-built from the schematics extracted from die photographs,
published at [og.kervella.org/sc01a](http://og.kervella.org/sc01a). The C++
port tracks [MAME's votrax.cpp](https://github.com/mamedev/mame/blob/master/src/devices/sound/votrax.cpp).
For deeper background, see `Tech overview.md`.

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

## Building the NVDA addon

From the repo root:

```
python nvda-addon/package.py
```

Output: `nvda-addon/votrax-<version>.nvda-addon` — drag-and-drop into NVDA to
install.

