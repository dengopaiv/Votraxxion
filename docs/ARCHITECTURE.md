# What is in this repository

A map of the parts and how they depend on each other. For the chip itself —
the history, the die analysis, the ROM encoding, the signal path — see
[tech-overview.md](tech-overview.md). This file is about the code.

## The one-paragraph version

There is one synthesizer, written in C, and everything else is a way of
driving it. It reproduces the Votrax SC-01 at the level of its analog signal
path: a 9-level glottal source and a noise LFSR through seven switched-
capacitor filters whose coefficients are rebuilt from the chip's own
interpolating parameter registers, sample by sample. Both mask ROMs and an
English letter-to-sound front end are compiled in, so the whole thing is one
library file with no data files beside it and no Python underneath it. That is
what lets the NVDA add-on be 142 KB.

## The parts

```
                      text ("Hello world.")
                            |
                   src/ttv.c  letter-to-sound
                            |     NRL rules -> ARPABET -> phone codes
                            v
                   packed phone bytes  (phone in bits 0-5, pitch in 6-7)
                            |
                   src/votrax.c  the scheduler
                            |     holds each phone for duration/speed samples
                            v
                   src/votrax_core.c  the chip
                            |     glottal + noise -> F1 F2v F2n F3 F4 FX
                            v
                      int16 mono samples at master_clock/18
```

| Path | What it is | Depends on |
|---|---|---|
| `src/` | **The synthesizer.** C11. The only libc it uses is `string.h`, `math.h` and one `calloc`/`free` pair for the chip handle; nothing allocates once audio is running. This is the deliverable. | nothing |
| `nvda-addon/` | The NVDA screen-reader driver: one Python shim over `src/` via ctypes, plus a packager that builds x64 and x86 libraries and zips a `.nvda-addon`. | `src/` |
| `gui-native/` | A one-window Win32 program over `src/`, statically linked: text in, a voice picked from the two masks and the clock/speed controls, audio out or a WAV saved. One ~200 KB executable, no DLL and no data file. Same shape as the native GUIs in the sibling SAM and STSPEECH projects. | `src/` |
| `csrc/` | pybind11 bindings exposing the chip core to Python as `pyvotrax._votrax_core`. | `src/`, pybind11 |
| `pyvotrax/` | The Workbench: a wxPython GUI, a TTS pipeline over CMUdict, preset handling. A music-production tool, not the screen-reader path. | `csrc/`, numpy, scipy, wx |
| `py_emu/` | The same DSP again, in pure Python, plus experimental "enhanced" modes (LF glottal, nasal anti-resonators) that the C core does not have. | numpy |
| `tests/` | 558 tests. Some drive Python, some load the built DLL through ctypes and skip if it is not built. | both |
| `tools/goldens.py` | Fingerprints a library's entire observable output so two implementations can be diffed sample-for-sample. Its committed output is `tests/data/golden.json`. | a built library |
| `tools/verify_gui.py`, `tools/verify_gui_keyboard.py` | Check the built GUI executable rather than a harness that shares its sources: the audio against the library over ctypes, and the tab order against the real window with posted keypresses. | a built GUI |
| `tests/test_datasheet.py`, `docs/DATASHEET.md` | The synthesizer against Votrax's 1980 data sheet: Table 1 durations, Table 2 categories in the ROM, the affricate rule, live inflection, the clock relation and knob, live clock changes and the Figure 8 output stage. | a built library |
| `tools/verify_rom.py` | Checks the three ROM transcriptions against each other, and against the mask ROM dumps when you supply them (they are not in the repository; see `reference/roms/README.md`). Runs inside the test suite too. | nothing; the dumps optional |
| `tools/compare_reference.py` | Diffs this engine against an independent build of MAME's device (the DLL in Tamas Geczy's `votraxsc01` add-on): audio per phone, phone-end timing, and which phones each front end ever emits. Takes the reference DLL as an argument; it is not vendored. | a reference DLL |
| `reference/` | Galibert's gate-level simulator and MAME's `votrax.cpp`. Provenance, not code we build. | nothing |
| `packaging/`, `presets/` | PyInstaller spec, Inno Setup script and factory presets for the Workbench. | `pyvotrax/` |

## Why `py_emu` still exists

It looks like a duplicate of the C core and mostly is. It earns its place as an
independent transcription: `py_emu/rom.py` decodes the ROM words in Python,
`src/votrax_rom.c` in C, and `reference/gate-sim/rom.cc` is Galibert's
original. Three transcriptions that agree is why the tables can be trusted, and
`tests/test_masks.py` checks two of them against each other on every run.

It is also where the experimental work lives — the LF glottal model and the
enhanced DSP modes are things the real chip does not do, so they do not belong
in a part whose job is to be the chip.

## The two things that are easy to get wrong

**Cancellation.** The chip latches a phone and voices it to completion. Dropping
the queue is not enough: without a reset, the phone in progress keeps sounding
and its remainder is heard at the head of the next utterance as a scrap of the
cancelled speech. `vx_cancel` resets and restores the inflection the reset
cleared.

**Rate.** There are two ways to speak faster and they are not interchangeable.
Moving the master clock is what the 1980 hardware's single knob did: tempo and
pitch rise together and the voice turns into a chipmunk. That is authentic and
useless at 400 words a minute. `vx_set_speed` instead truncates — each phone is
held for its natural length divided by `speed` and the next is committed early,
so the formant interpolators carry on toward the new targets from wherever they
had reached. Tempo moves, pitch does not, and the audio is still the chip's own
output with no resampling anywhere.

## A property worth knowing

The chip's *sample values* are almost entirely independent of the master clock.
`sclock` and `cclock` are `master/18` and `master/36`, so they scale together
and the bilinear-transformed filter coefficients come out invariant; changing
the clock is realized as a different playback rate for the same waveform. The
one exception is the noise-shaper filter, whose `k1` has `cclock` in the
numerator where the others have it in the denominator, so the fricative path
does change with the clock. Both behaviours are inherited from MAME's die
analysis and are reproduced deliberately.

## Building

The synthesizer alone, which is all the NVDA add-on needs:

```
cd nvda-addon && python package.py       # both architectures + the .nvda-addon
```

The Python extension for the Workbench:

```
pip install -e .[gui,dev]
python setup.py build_ext --inplace
python -m pytest tests/
```

See [../README.md](../README.md) for the Workbench app and installer.
