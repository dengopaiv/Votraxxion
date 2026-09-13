# Reference material

Nothing here is built, shipped, or imported. It is kept because it is the
provenance of the tables in the synthesizer, and a claim about where ROM data
came from is worth nothing if the source is only in a commit message.

## `roms/`

Where the two dumped 512-byte mask ROMs, `sc01.bin` and `sc01a.bin`, go if you
have them — the only material that is the chip itself rather than someone's
reading of it. **They are not in the repository** and must not be committed;
`roms/README.md` says why, and how to supply your own. With them,
`tools/verify_rom.py` checks every transcription below against the silicon, and
the test suite runs that check; without them it checks the transcriptions
against each other.

## `gate-sim/`

Olivier Galibert's gate-level simulator of the SC-01-A, written from his own
die photographs of the part (published at
[og.kervella.org/sc01a](http://og.kervella.org/sc01a)).

`rom.cc` is the one that matters: it holds the 64 rows of 12-bit `word0` and
32-bit `word1` transcribed off the die, and the bit-extraction that turns them
into the twelve phoneme parameters. `src/votrax_rom.c` carries the same numbers
and the same extraction, and `py_emu/rom.py` a third copy in Python — three
independent transcriptions that agree, which is why the tables can be trusted.
All three agree with the SC-01-A dump as well, row for row — checked on
2026-09-10 and on every run where the dumps are supplied — which is why they
can be trusted against the chip rather than just against each other.

The rest (`blocks.cc`, `sched.cc`, `sram.cc`, `vsim.cc`, `state.h`) simulate
the chip's digital half gate by gate. That is a slower and more literal thing
than this project needs — the synthesizer reproduces the same behaviour from
the decoded parameters directly — but it is the ground truth the timing
engine was checked against.

Build it with the `Makefile` here if you want to run it; it needs g++ and
nothing else.

## `mame_votrax.cpp`

MAME's `src/devices/sound/votrax.cpp`, also Galibert's. The analog signal path
in `src/votrax_core.c` tracks this file: the filter topologies, the capacitor
values, the 9-level glottal waveform, the noise LFSR and the closure
attenuation curve are all his analysis of the die, and the deliberate
divergences are listed in `docs/tech-overview.md`, Part 2, "Key differences
from MAME".

BSD-3-Clause, as MAME is.

## `SC-01_Data_Sheet_v1_text.pdf`

Votrax's own SC-01 data sheet (Troy, MI; copyright 1980), a scan with an OCR
text layer. It is the manufacturer's statement of the chip's behaviour rather
than an analysis of the die, so it is the one independent check on the
numbers above. Pages 1–9 are the data sheet proper; page 10 is a later sheet
of application circuits for the SC-01A. The copy has handwritten annotations
(the phoneme-category letters in Table 1 and column notes) that are a
previous owner's, not Votrax's.

What in it bears on the synthesizer:

- **Table 1** — the 64 phonemes with code, symbol and duration in ms at the
  nominal 720 kHz clock; **Table 2** — the same phonemes by production
  category (voiced, voiced fricative, voiced stop, fricative stop, fricative,
  nasal, no sound).
- The front-end rules under Table 1: T must precede CH, and D must precede J.
- The master clock: nominally 720 kHz, `f ≈ 1.25 / RC`; Figures 6 and 7 show
  voice variation by a potentiometer (6.8 kΩ + 50 kΩ audio taper, 120 pF on
  page 10) and by DAC current injection.
- I1/I2 set the pitch level of voiced phonemes instantaneously; A/R requests
  may be ignored for external phoneme timing.
- Figure 8 on page 10 — a reference output stage through an LM386.

Each of these has been checked against the synthesizer or implemented in it,
and `docs/DATASHEET.md` records which, how, and what was measured.

The sheet states: "Rights for the reproduction and distribution of the data
contained herein are granted except for the manufacture and reproduction of
the subject equipment."
