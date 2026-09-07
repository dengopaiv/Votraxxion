# The C rewrite

## Why

The synthesizer works and is already native: `csrc/votrax_capi.cpp` compiles to
a 195 KB DLL that the NVDA add-on drives over ctypes, with both mask ROMs and
the English front end inside it. The rewrite is not about capability. It is
about what the library *is*.

Today it is C++ — header-only classes, `std::string`, `std::vector`,
`std::deque`, templates, `new`. That costs four things that matter for a screen
reader:

1. **A C++ runtime dependency.** The DLL links `MSVCP140.dll`. On a machine
   without the Visual C++ redistributable the add-on does not load, and the
   failure mode is NVDA silently not listing the synthesizer — no error a user
   can act on.
2. **One toolchain.** MSVC-only in practice. Plain C builds under MinGW, clang,
   or gcc for a Linux screen reader, with no ABI questions.
3. **Allocation on the speech path.** `std::deque` allocates when phones are
   queued and `ttv::translate` builds `std::string`s and `std::vector`s per
   utterance. It is fast enough today, but a synthesizer feeding an audio
   callback should not be taking the allocator lock at all.
4. **Size.** Most of the 195 KB is template instantiation and exception
   tables for a program that never throws.

C also matches what the thing is. The chip is a state machine and seven IIR
filters over a fixed set of arrays. None of that wanted a class.

## The rule

**Byte-identical output, verified, or it is a bug.** A DSP rewrite that
"sounds the same" is a rewrite nobody can trust. `tools/goldens.py` captures a
fingerprint of everything the library can be asked to do — every one of the 64
phones on both masks, whole utterances at five speeds and three clocks, the
front end on text covering numbers and abbreviations and all three sentence
contours, cancel and reset, the ready handshake — and diffs two libraries
entry by entry. The C++ baseline is captured before the first line of C is
written. The C library must match it exactly.

The one licensed exception: `ttv_*` output is byte-identical too, but if the C
front end ever needs to differ, that is a decision to write down here, not a
diff to accept quietly.

## The shape

`src/`, C11, one translation unit per concern, each with a header:

| File | From | Notes |
|---|---|---|
| `votrax_rom.{h,c}` | `csrc/rom_data.h` | Both mask tables, decoded once at load. `enum class` → `enum`, `std::array` → array. Mechanical. |
| `votrax_filters.{h,c}` | `csrc/filters.h` | Already C but for `<cmath>` and two templates. The templates become fixed-size loops. Mechanical. |
| `votrax_core.{h,c}` | `csrc/votrax_core.h` | The class becomes a struct plus functions taking `vx_core *`. All state is already plain scalars and arrays. Mechanical, but this is the file where a typo is a different voice, so it goes one method at a time against the goldens. |
| `ttv_tables.{h,c}` | `csrc/ttv_tables.h` | `const char *` tables. Mechanical. |
| `ttv.{h,c}` | `csrc/ttv.h` | **The real work.** `std::string`/`std::vector` throughout. Becomes fixed-capacity buffers with explicit bounds, which is also where the current code's silent truncation behaviour has to be reproduced exactly. |
| `votrax.{h,c}` | `csrc/votrax_capi.{h,cpp}` | The public API, unchanged in signature. `std::deque` becomes a ring buffer. |

`csrc/` keeps only `bindings.cpp`, which includes the C headers through
`extern "C"` and stops being a second copy of anything.

## The order

Bottom-up, so that each step is verifiable before the next depends on it.

- [x] **0. Baseline.** Build the C++ DLL, capture goldens. *(done — 42 entries; 60/64 phone hashes unique, exactly 12 phones differ between masks, matching the 12 documented ROM deltas.)*
- [x] **1. Cleanup.** Remove build output and the superseded Python add-on, move reference material out of the root. *(done — 560 MB to 2.1 MB.)*
- [x] **2. Documentation.** [ARCHITECTURE.md](ARCHITECTURE.md), this file.
- [ ] **3. `votrax_rom.c`** — tables and decode, checked against the C++ decode for all 64 phones on both masks.
- [ ] **4. `votrax_filters.c`** — coefficient builders, checked coefficient-by-coefficient against the C++ ones at three clocks.
- [ ] **5. `votrax_core.c`** — the chip. Checked by rendering every phone and diffing samples.
- [ ] **6. `ttv_tables.c` + `ttv.c`** — the front end. Checked by translating a word list far larger than the goldens' corpus and diffing phone streams.
- [ ] **7. `votrax.c`** — API and scheduler. Full golden comparison passes.
- [ ] **8. Wire it up.** `nvda-addon/package.py` builds C; `bindings.cpp` includes the C headers; `csrc/` loses its duplicates.
- [ ] **9. Regression.** Goldens identical, 500 existing tests green, a test that keeps the goldens honest from here on.

## Step log

Kept short on purpose: what changed, and what proved it.

**0. Baseline** — Built `csrc/votrax_capi.cpp` with MSVC 19 (VS 18 Community):
195 KB. Captured 42 golden entries in 1.75 s.

Two things surfaced while making the harness discriminate:

- A freshly reset chip is *silent for its first ~1200 samples* while the
  formant interpolators ramp from zero. The first version of the live-switch
  check hashed 1024 samples and so compared silence to silence, passing
  regardless. Windows are 4096 samples now.
- Changing the master clock does not change the sample values, except through
  the noise path. `sclock` and `cclock` scale together and the bilinear
  transform normalizes them out; only `build_noise_shaper_filter` has `cclock`
  in a numerator, so only the fricatives move. The corpus keeps all three
  clocks because that asymmetry is exactly the kind of thing a rewrite breaks.

**1. Cleanup** — 560 MB to 2.1 MB. The old Python add-on carried 299 MB of
vendored numpy/scipy/CMUdict wheels to do what the native one does in 169 KB.
Galibert's gate-level simulator and MAME's `votrax.cpp` moved to `reference/`
with a README saying what each is evidence for, rather than being deleted:
`rom.cc` is where the ROM tables came from and that claim needs a source.

**2. Documentation** — This file and `ARCHITECTURE.md`. The architecture doc
records the two things that are easy to get wrong (cancellation leaving a tail,
and the two incompatible meanings of "faster") because both were learned the
expensive way and neither is visible in the code.
