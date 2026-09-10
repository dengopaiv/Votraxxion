# The two SC-01 mask ROMs

The 512 bytes on the die, for each of the two production masks. Everything else
in this repository that claims to know what a phoneme sounds like is downstream
of these two files.

| File | Size | CRC32 | SHA-1 |
|---|---|---|---|
| `sc01.bin` | 512 | `528d1c57` | `268b5884dce04e49e2376df3e2dc82e852b708c1` |
| `sc01a.bin` | 512 | `fc416227` | `1d6da90b1807a01b5e186ef08476119a862b5e6d` |

Those are the same hashes as the `ROM_LOAD` lines in `../mame_votrax.cpp`, so
these are the standard dumps and not somebody's re-derivation.

## Why they are here

The phoneme tables exist four times over: Galibert's die transcription in
`../gate-sim/rom.cc`, C arrays in `src/votrax_rom.c`, Python tuples in
`py_emu/rom.py`, and these dumps. The first three are people reading numbers
off a photograph, and three transcriptions of the same typo would agree with
each other perfectly. Only the dumps settle it against the chip.

`python tools/verify_rom.py` checks all four against each other and prints what
it found; `tests/test_masks.py::TestAgainstTheDumps` runs the same check on
every test run, and additionally decodes both dumps and compares the twelve
parameters against what the built extension returns. A future edit to any table
now has to survive the silicon.

## What is in them

Each row is a little-endian 64-bit word:

```
bits  0-31   word1 -- the seven 4-bit parameters, interleaved
bits 32-43   word0 -- closure, duration, and the top of cld/vd
bits 44-55   unused, zero in both masks
bits 56-61   the phone number this row is for
```

The ROM is **content-addressed**: a row carries its own phone number and the
rows are not in phone order — `sc01a.bin` starts 03, 3E, 3F, 3D, 3C, 3B. That
is why MAME scans all 64 rows looking for a match instead of indexing, and it
is the first thing to get wrong when reading these files.

The two masks differ in exactly twelve rows — 08, 13, 15, 23, 24, 2E, 2F, 30,
31, 32, 33, 3D — and in exactly one field, the voice amplitude `va`. word0 is
identical throughout. All twelve are open vowels, and the 1980 part ran every
one of them at full scale; the -A revision pulled them down to 9, 11 or 14.
See `docs/tech-overview.md`, Part 1, "The two mask revisions".

## Provenance and status

Extracted from `votraxsc01-1.0.2.nvda-addon`, a third-party NVDA add-on by
Tamas Geczy that ships MAME's SC-01 device with the dumps beside it; the same
files are in MAME and have been archived openly for years. They are the
contents of a commercial IC whose maker stopped producing it in the late 1980s.

They are kept here as reference material with the provenance written down, the
same way `gate-sim/` and `mame_votrax.cpp` are. Nothing builds against them and
nothing ships them: the synthesizer compiles its tables in, so `src/` has no
ROM file to load at runtime and the NVDA add-on has none to carry.
