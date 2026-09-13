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

## Tamas Geczy — the NVDA driver, the phoneme map, the exception dictionary

His work is in four places here, all from his `votraxsc01` NVDA add-on, which
wraps MAME's SC-01 device: the driver, two front-end tables, the scheduler's
rate and cancel design, and a handful of later fixes and tests.

**The driver.** `nvda-addon/addon/synthDrivers/votraxNative.py` is derived from the driver in
Geczy's `votraxsc01` NVDA add-on
([github.com/tgeczy/votraxsc01-nvda](https://github.com/tgeczy/votraxsc01-nvda)),
version 1.0.2, which wraps MAME's SC-01 device. BSD-3-Clause, copyright holder
tgeczy. What came from it: the shape of the driver — one speak thread that
alone touches the chip, a queue of work items, cancellation by epoch counter
with control items exempt, audio fed in ~12 ms blocks with the epoch
re-checked at the feed — the constant-pitch rate by phone truncation and the
"Authentic rate" setting that makes rate the master clock, the pitch setting
snapped onto the chip's four inflection levels, and the utterance close of
STOP plus a rendered tail. Many of its comments survive word for word.

What changed here: the chip, the ROM tables and the letter-to-sound matcher
are this repository's native C rather than MAME's device and a ROM file, so
the ROM search, migration and checksum are gone; and the sentence contour is
new.

**The scheduler's design.** Rate by phone truncation, with the clock kept as
"authentic rate", and the rule that a cancel must reset the chip rather than
only drop the queue (his fix for the previous utterance leaking into the next)
are his. Here they live in the C scheduler (`vx_set_speed`, `vx_speak`,
`vx_cancel` in `src/votrax.c`), which works from the exact phone length rather
than measuring each phone at startup; the implementation is ours.

**Two front-end tables** in `src/ttv_tables.c`, first recovered from his
`sc01.dll` and since checked entry for entry against his source:

- `TTV_ARPABET` — his transcription (`arpabet_to_sc01.c`) of NRL Report 7948's
  IPA-to-Votrax translation rules from the report's SNOBOL listing, with his
  two documented repairs of typos that every surviving transcription shares.
  The rules are public domain; the transcription is his. 81 entries, identical
  and in the same order.
- `TTV_EXCEPTIONS` — his exception dictionary (`exceptions.c`): respellings of
  words he measured broken through the NRL rules and measured correct after.
  17 entries, identical.

The symbol normalization `src/ttv.c` applies before that map (upper-casing,
`j` to `JH`, `NG` to `NX`) is the one his map expects.

**Later fixes and tests**, recorded in `docs/REWRITE.md`: the corrected letter
names for O, U and S (his changelog found them; H, Q, W and Y were found
checking the rest); the cancel-under-load, chip-silence and truncation-rate
tests in `tests/test_nvda_driver.py`, adapted from his
`tests/driver_cancel_test.py`; and the modes of `votrax-say` (phone strings, the
64-phone table, the name list), which follow his `say01` probe — the code is
ours.

`tools/compare_reference.py` diffs this engine against his add-on's DLL, and
the ROM dumps in `reference/roms/` were extracted from his add-on (they are
not his work, and are covered below). His notice is the second line of
`LICENSE`, and the full repository is held locally at
`reference repositories/votraxsc01-nvda`.

This credit was missing from release 1.1.0 and earlier, and the tables were
attributed there only to "a third-party NVDA driver"; it was added on
2026-09-13.

## US Naval Research Laboratory and John A. Wasser — the letter-to-sound rules

The rules behind `src/ttv_tables.c` are from Elovitz, Johnson, McHugh and
Shore, *Automatic Translation of English Text to Phonetics by Means of
Letter-to-Sound Rules*, NRL Report 7948 (1976). A work of the US Government,
in the public domain. That covers both stages: the letter-to-sound rules and
the IPA-to-Votrax rules, whose transcription is Geczy's (above).

`TTV_NRL_RULES` is John A. Wasser's 1985 public-domain `english.c`, the shape
the rules circulated in with Votrax-era hardware: its 350 letter rules are
identical and in order. `TTV_CARDINALS`, `TTV_ORDINALS`, `TTV_ASCII_NAMES`
and `TTV_ABBREVIATIONS` are from his `saynum.c`, `spellword.c` and `parse.c`
in the same posting. The number reader in `src/ttv.c` follows the shape of
`saynum.c`, with the departures listed in `docs/REWRITE.md`.

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

## Votrax — the SC-01 data sheet

`reference/SC-01_Data_Sheet_v1_text.pdf` is Votrax's data sheet for the chip,
copyright Votrax 1980. It is redistributed under the grant printed on its first
page: "Rights for the reproduction and distribution of the data contained
herein are granted except for the manufacture and reproduction of the subject
equipment." It is not shipped; see `reference/README.md`.

## Päiv Dengo — the work in this repository

The C synthesizer's structure and the scheduler's implementation
(`src/votrax.c`, with the rate and cancel design credited to Geczy above), the
English front end's matcher, number reader and prosody (`src/ttv.c`; its
tables are credited above), the flat C API, the changes to the NVDA driver
described above, `votrax-say`, the native GUI,
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
