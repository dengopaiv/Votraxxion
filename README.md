# Votrax SC-01

An emulator of the Votrax SC-01 speech synthesizer chip, in C, packaged as an
NVDA screen-reader add-on and as a Windows music-production workbench.

The synthesizer is C with no dependencies -- not on Python, not on a phoneme
dictionary, not on a ROM file. Both mask ROMs and the whole English front end
are compiled in, so the NVDA add-on is one Python shim and one 153 KB library
per architecture, and nothing else.

The DSP is hand-built from the schematics extracted from die photographs,
published at [og.kervella.org/sc01a](http://og.kervella.org/sc01a), and tracks
[MAME's votrax.cpp](https://github.com/mamedev/mame/blob/master/src/devices/sound/votrax.cpp).
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the pieces fit
together and [docs/tech-overview.md](docs/tech-overview.md) for the chip
itself. [docs/ROADMAP.md](docs/ROADMAP.md) puts it in context: the SC-01 is one
of three Votrax sound engines, and the other two — the SC-02/SSI-263 and the
discrete VS-6/ML-1 that came before both — are separate synthesizers with
separate libraries and separate add-ons. This one is the SC-01, and only the
SC-01.

## Running from source

Prerequisites:

- Python 3.9+
- A compiler toolchain for the pybind11 extension (on Windows: MSVC via
  Visual Studio Build Tools).

Install and build:

```
pip install -e .[gui,dev]
python setup.py build_ext --inplace
python -m pytest tests/
python -m pyvotrax
```

`setup.py` builds the pybind11 shim in `csrc/` against the C core in `src/`.
The NVDA add-on does not go through it — see below.

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

## Building the native GUI

`gui-native/` is a one-window Win32 program over the same C sources: type
something, pick a voice, hear it, save it. It is the same shape as the native
GUIs in the sibling SAM and STSPEECH projects, and unlike them it has nothing
to bundle — both mask ROMs and the English front end are already inside the
synthesizer, so the whole application is one executable of about 200 KB with a
static CRT and no data file beside it.

```
gui-native\build.cmd          # x64
gui-native\build.cmd x86      # and the 32-bit build
```

Output: `gui-native/build/votrax_gui-x64.exe`. The script finds MSVC itself
through `vswhere`; there is nothing to generate first.

What the window offers, beyond text in and audio out:

- **Both mask revisions** as a voice choice — the 1980 SC-01 and the SC-01-A.
- **Clock and speed as separate controls**, because they are separate things.
  Moving the master clock is the 1980 hardware's single knob: tempo and pitch
  rise together and the sample rate moves with them, 40 kHz at the datasheet
  720 kHz and 60 kHz at the 1.08 MHz "chipmunk" preset. Speed truncates each
  phone instead and leaves the clock alone, so tempo moves and pitch does not.
- **Phoneme mode**, taking datasheet names — `H AH1 L OO PA1` — with an
  optional `:0` to `:3` for per-phone pitch. **Convert to Phonemes** turns the
  text box into exactly that notation, so what comes back can be edited and
  spoken again.
- **Voice presets** over the four things the chip actually has, including the
  clock figures the Workbench's factory presets already use for Chipmunk and
  Slow robot.

Every control is a standard Win32 one with a label before it in tab order and
an `&` accelerator, which is what a screen reader expects. Two tools check that
this is true of the built binary rather than merely intended:

```
python tools/verify_gui.py            # audio, against the library itself
python tools/verify_gui_keyboard.py   # the keyboard, against the real window
```

`verify_gui.py` drives `votrax_gui.exe --selftest` through 22 cases and compares
the WAV bytes with the same work done through the shipped DLL over ctypes — the
executable contains a second build of the engine, and this is what says the two
agree. `verify_gui_keyboard.py` launches the real window, posts actual Tab
keypresses into its queue and reads back where the focus went, which is how a
keyboard trap in the multiline text box gets caught instead of argued about.

## Building the standalone synthesizer

`src/votrax.h` is the C API; `docs/tech-overview.md`, Part 4, has the details.

Windows, MSVC (from a Developer Command Prompt). Compile from inside `src/`:
MSVC's `/Fo` will not take a quoted path ending in a backslash, so a repository
path with a space in it breaks the obvious command line.

```
cd src
cl /std:c11 /O2 /LD /I. /Fe:votraxsc01.dll votrax.c votrax_core.c votrax_filters.c votrax_rom.c ttv.c ttv_tables.c
```

Linux or macOS:

```
cc -std=c11 -O2 -shared -fPIC -Isrc -o libvotraxsc01.so src/*.c
```

Driving it is the loop any Votrax front end has always used — turn text into
phones, queue them, and pull audio until the queue drains:

```python
import ctypes, wave

lib = ctypes.CDLL("./votraxsc01.dll")
u8p = ctypes.POINTER(ctypes.c_ubyte)
for name, args, ret in [
    ("vx_create",      [ctypes.c_int, ctypes.c_uint],              ctypes.c_void_p),
    ("vx_destroy",     [ctypes.c_void_p],                          None),
    ("vx_sample_rate", [ctypes.c_void_p],                          ctypes.c_double),
    ("vx_speak",       [ctypes.c_void_p, u8p, ctypes.c_int],       ctypes.c_int),
    ("vx_pending",     [ctypes.c_void_p],                          ctypes.c_int),
    ("vx_render",      [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int], ctypes.c_int),
    ("ttv_translate",  [ctypes.c_char_p, u8p, ctypes.c_int],       ctypes.c_int),
]:
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = args, ret

chip = lib.vx_create(1, 0)          # 1 = the 1980 SC-01 mask, 0 = default clock

phones = (ctypes.c_ubyte * 4096)()
n = lib.ttv_translate(b"Hello world.", phones, 4096)
lib.vx_speak(chip, phones, n)

BLOCK = 512
audio, block = bytearray(), (ctypes.c_int16 * BLOCK)()

def pull():
    # Never ask for more samples than the buffer holds: vx_render writes
    # exactly what it is told to and does not know how big the buffer is.
    lib.vx_render(chip, block, BLOCK)
    audio.extend(memoryview(block).cast("B"))

while lib.vx_pending(chip):
    pull()
for _ in range(8):                  # the last phone is still sounding
    pull()

with wave.open("hello.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(int(lib.vx_sample_rate(chip)))
    w.writeframes(audio)

lib.vx_destroy(chip)
```

Declare every `argtypes` before calling: without them ctypes guesses, and a
guessed 32-bit handle in a 64-bit process is a crash that only shows up once
the heap wanders past 4 GB.

## Building the NVDA addon

The add-on is one Python file and one DLL per architecture — 142 KB in total,
with no bundled wheels, no pronunciation dictionary and no ROM files:

```
cd nvda-addon
python package.py
```

That builds both libraries (the script drives MSVC and finds vcvars itself; the
sources are plain C11 and build under MinGW or clang too) and writes
`votraxsc01-1.0.0.nvda-addon`. NVDA 2026 is 64-bit only, so the x64
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

There was an earlier driver that ran the pyvotrax emulator with numpy, scipy
and CMUdict bundled alongside it, about 40 MB in all. The native add-on does
the same job in 142 KB and replaced it; it is in the git history if you want to
see it.
