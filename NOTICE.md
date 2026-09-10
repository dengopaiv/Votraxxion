# Notices and attributions

Who made what in this repository, and under what terms. The whole thing is
BSD-3-Clause (`LICENSE`), which it can be because the upstream it descends
from is too — but "BSD-3-Clause" on its own does not say whose copyright
notice has to survive, and that is what this file is for.

## Olivier Galibert — the die analysis

The part of this that could not have been guessed. Galibert decapped an
SC-01-A, photographed the die and worked out what was on it: the ROM contents
and their bit interleaving, the capacitor values of the switched-capacitor
filters, the nine-level glottal ladder, the noise LFSR polynomial, the
interpolation and timing structure. Published in MAME from 0.181 (2016), with
the schematics at [og.kervella.org/sc01a](https://og.kervella.org/sc01a/).

Everything in `src/votrax_core.c`, `src/votrax_filters.c` and
`src/votrax_rom.c` tracks that work. `reference/mame_votrax.cpp` and
`reference/gate-sim/` are his code, kept as provenance rather than built.
BSD-3-Clause; his copyright notice is the first line of `LICENSE` and stays
there.

Where this repository deliberately diverges from his implementation — the
F2-noise injection filter, which MAME neutralizes and this does not — it is
recorded in `docs/tech-overview.md`, Part 2, with the difference measured
rather than asserted.

## US Naval Research Laboratory — the letter-to-sound rules

`src/ttv_tables.c` is the ruleset from Elovitz, Johnson, McHugh and Shore,
*Automatic Translation of English Text to Phonetics by Means of Letter-to-Sound
Rules*, NRL Report 7948 (1976). A work of the US Government, in the public
domain. The arrangement follows John A. Wasser's 1985 public-domain C version,
which is the shape the tables circulated in with Votrax-era hardware.

The punctuation timing is not theirs: NRL mapped every mark to a space,
because it was a letter-to-sound algorithm with no opinions about timing. What
a comma is worth here is a decision made in this repository, recorded in
`docs/REWRITE.md`.

## The mask ROMs

`reference/roms/sc01.bin` and `sc01a.bin` are the contents of a commercial
integrated circuit whose maker stopped producing it in the late 1980s. They
are the standard dumps, archived openly in MAME and elsewhere, and they are
kept here as reference material with their provenance written down —
`reference/roms/README.md`. No claim of ownership is made over them and
nothing here licenses them to anyone.

They are not shipped. The synthesizer compiles its tables in, so neither the
library nor the add-on carries a ROM file.

## Päiv Dengo — the work in this repository

The C synthesizer's structure and scheduler (`src/votrax.c`, the rate and
cancellation behaviour), the English front end (`src/ttv.c` and the
arrangement of its tables), the flat C API, the NVDA driver, the native GUI,
the verification tooling and the packaging. Written with Claude (Anthropic),
session by session; the commit trailers record which.

That last point is worth being plain about rather than coy: an AI assistant is
not a copyright holder and Anthropic does not claim one in what Claude
produces, so the copyright line is a person's. The credit for the collaboration
belongs in the commit history, where it is, not in `LICENSE`.

## For the Workbench only

`pyvotrax/` uses the CMU Pronouncing Dictionary (Carnegie Mellon University,
BSD-style) and, when built as an application, PyInstaller-bundled numpy, scipy,
wxPython and sounddevice under their own licences. None of that reaches the
native library, the NVDA add-on or the GUI, which is most of why those are
measured in kilobytes.

## Patents

Gagnon's 3,836,717 (1974) and 3,908,085 (1975), and the switched-capacitor
filter patent 4,433,210, have all long expired.
