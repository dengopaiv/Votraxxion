# Reference material

Nothing here is built, shipped, or imported. It is kept because it is the
provenance of the tables in the synthesizer, and a claim about where ROM data
came from is worth nothing if the source is only in a commit message.

## `gate-sim/`

Olivier Galibert's gate-level simulator of the SC-01-A, written from his own
die photographs of the part (published at
[og.kervella.org/sc01a](http://og.kervella.org/sc01a)).

`rom.cc` is the one that matters: it holds the 64 rows of 12-bit `word0` and
32-bit `word1` transcribed off the die, and the bit-extraction that turns them
into the twelve phoneme parameters. `src/votrax_rom.c` carries the same numbers
and the same extraction, and `py_emu/rom.py` a third copy in Python — three
independent transcriptions that agree, which is why the tables can be trusted.

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
