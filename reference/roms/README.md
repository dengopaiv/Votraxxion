# The two SC-01 mask ROMs — supply your own

The 512 bytes on the die, for each of the two production masks. **They are not
in this repository, and must not be added to it.** They are the contents of a
commercial chip, not anyone's work here, and although the copyright in them is
almost certainly ownerless after four decades, it is uncleared. Tamas Geczy's
`votraxsc01-nvda` follows the same rule for the same reason: his repository never
carries the dumps, although his release bundles do.

Nothing that builds or ships from this repository needs them. The synthesizer
compiles its phoneme tables in (`src/votrax_rom.c`), and those tables are
Galibert's transcription of the die, published in MAME's source under BSD-3.
The dumps are needed for two checks only:

- `tools/verify_rom.py` and `tests/test_masks.py::TestAgainstTheDumps`, which
  prove the tables against the silicon;
- `tools/compare_reference.py`, whose reference engine loads them.

Without them, `verify_rom.py` still checks the three transcriptions against
each other and says it did not check them against the chip, and the dump tests
skip with a reason.

## Supplying them

They are the standard MAME files, present in any MAME set for a machine that
carries the chip (`votrtnt`, `votrpss`, `gorf`, `qbert`, and others):

| File | Size | CRC32 | SHA-1 |
|---|---|---|---|
| `sc01.bin` | 512 | `528d1c57` | `268b5884dce04e49e2376df3e2dc82e852b708c1` |
| `sc01a.bin` | 512 | `fc416227` | `1d6da90b1807a01b5e186ef08476119a862b5e6d` |

Those are the hashes in the `ROM_LOAD` lines of `../mame_votrax.cpp`. Either
place both files in this folder, where `.gitignore` keeps every `*.bin` out of
commits, or keep them anywhere else and point `VOTRAX_ROM_DIR` at that folder:

```
set VOTRAX_ROM_DIR=C:\path\to\your\roms
python tools/verify_rom.py
```

`verify_rom.py` refuses a file whose CRC32 or SHA-1 differs, so a wrong dump
cannot pass as the right one.

## History

From 2026-09-10 to 2026-09-13 the two files were committed here, in
`reference/roms/`. They were removed from every commit of the repository's
history on 2026-09-13 with `git filter-repo`, and the release tags were
recreated on the rewritten commits. The author keeps a private copy outside
any repository.

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
is the first thing to get wrong when reading these files. It also means an
image rebuilt from the tables in phone order drives MAME's device identically,
though its CRC will not match the dump's.

The two masks differ in exactly twelve rows — 08, 13, 15, 23, 24, 2E, 2F, 30,
31, 32, 33, 3D — and in exactly one field, the voice amplitude `va`. word0 is
identical throughout. All twelve are open vowels, and the 1980 part ran every
one of them at full scale; the -A revision pulled them down to 9, 11 or 14.
See `docs/tech-overview.md`, Part 1, "The two mask revisions".

## Provenance

The copies this repository was verified against came from
`votraxsc01-1.0.2.nvda-addon`, Tamas Geczy's NVDA add-on, whose release bundle
ships MAME's SC-01 device with the dumps beside it. The same files are in MAME
and have been archived openly for years. They are the contents of a commercial
IC whose maker stopped producing it in the late 1980s, read optically from
decapped dies by the MAME project.
