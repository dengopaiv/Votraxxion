#!/usr/bin/env python3
"""Diff this engine against an independent Votrax SC-01 implementation.

    python tools/compare_reference.py <path-to-reference-dll> [rom-dir]

The reference is any DLL exposing MAME's device behind a `vx_*` C API that
takes the mask ROM as a buffer:

    vx_create(variant, clock_hz, rom_bytes, rom_len, err_buf, err_len)

`votraxsc01-1.0.2.nvda-addon` (Tamas Geczy) ships one, which is where this
was developed. That binary is not vendored here -- it is somebody else's
build of somebody else's code -- so its path is an argument. Its engine
needs the mask ROM dumps, which are not in this repository: the ROM folder
is the second argument, else VOTRAX_ROM_DIR, else `reference/roms/` if you
have placed your own copies there (see its README).

**Its variant numbering is the inverse of ours.** There, 0 is the 1980
SC-01 and 1 is the SC-01-A; here `VX_MASK_SC01A` is 0. Getting that
backwards makes every phone differ for a reason that has nothing to do
with the engines, which is worth knowing before reading any output.

Three things are compared, and they answer different questions:

  * **Audio**, with both engines held on each phone for an identical
    fixed count of samples, so the state advance is identical and only
    the arithmetic is under test.
  * **Phone-end timing**, polled one sample at a time, which is what
    `vx_ready` promises and what a scheduler is built on.
  * **Front ends**, over a corpus, as a histogram of which phones each
    one ever emits. A phone the translator never produces is a phone the
    voice does not have, however well the DSP renders it -- that is how
    the missing PA1 was found (see docs/REWRITE.md).

Nothing here asserts that the reference is right. It is one more
independent transcription of the same die analysis, and where the two
disagree the disagreement is the finding.
"""
import collections
import ctypes
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import goldens  # noqa: E402

OURS = os.path.join(ROOT, 'nvda-addon', 'addon', 'synthDrivers',
                    'votraxNative-x64.dll')
DEFAULT_ROMS = os.environ.get('VOTRAX_ROM_DIR') or os.path.join(ROOT, 'reference', 'roms')

HOLD = 8000          # samples held per phone, identical for both engines
TAIL = 4000

CORPUS = [
    'The quick brown fox jumps over the lazy dog.',
    'She sells sea shells by the sea shore.',
    'Peter Piper picked a peck of pickled peppers.',
    'This measure is a pleasure; vision, azure, treasure.',
    'Judge Judy juggled generous gingerbread.',
    'Thirty three thousand feathers on a thrush is throat.',
    'Which witch wished which wicked wish?',
    'Doctor Smith paid forty two dollars on January third.',
    'Congratulations! What? Really. Now is the time.',
    'Sing song ringing longing. Thin thing, then those.',
]


class Ours:
    label = 'ours'

    def __init__(self, mask):
        self.lib = goldens.bind(OURS)
        self.chip = self.lib.vx_create(mask, 0)

    def close(self):
        self.lib.vx_destroy(self.chip)


class Reference:
    label = 'reference'

    def __init__(self, dll, rom_dir, mask):
        lib = ctypes.CDLL(dll)
        p = ctypes.c_void_p
        lib.vx_create.restype = p
        lib.vx_create.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_char_p,
                                  ctypes.c_uint, ctypes.c_char_p,
                                  ctypes.c_size_t]
        for name, res, args in (
                ('vx_destroy', None, [p]), ('vx_reset', None, [p]),
                ('vx_sample_rate', ctypes.c_double, [p]),
                ('vx_write', None, [p, ctypes.c_ubyte]),
                ('vx_inflection', None, [p, ctypes.c_ubyte]),
                ('vx_ready', ctypes.c_int, [p]),
                ('vx_render', ctypes.c_int,
                 [p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int]),
                ('ttv_translate', ctypes.c_int,
                 [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int])):
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
        self.lib = lib

        # Their variant numbering is the inverse of ours.
        rom_name = 'sc01a.bin' if mask == 0 else 'sc01.bin'
        with open(os.path.join(rom_dir, rom_name), 'rb') as f:
            rom = f.read()
        err = ctypes.create_string_buffer(256)
        self.chip = lib.vx_create(1 - mask, 720000, rom, len(rom), err, 256)
        if not self.chip:
            raise RuntimeError(err.value.decode('ascii', 'replace'))

    def close(self):
        self.lib.vx_destroy(self.chip)


def play(engine, phones, inflection=1):
    """Write each phone and hold it for exactly HOLD samples."""
    lib, chip = engine.lib, engine.chip
    total = HOLD * len(phones) + TAIL
    buf = (ctypes.c_int16 * total)()
    lib.vx_reset(chip)
    lib.vx_inflection(chip, inflection)
    at = 0
    for ph in phones:
        lib.vx_write(chip, ph)
        lib.vx_render(chip, (ctypes.c_int16 * HOLD).from_buffer(buf, at * 2),
                      HOLD)
        at += HOLD
    lib.vx_render(chip, (ctypes.c_int16 * TAIL).from_buffer(buf, at * 2), TAIL)
    return list(buf)


def ready_at(engine, phone, inflection=1, limit=120000):
    """Exact samples before the chip asks for the next phone."""
    lib, chip = engine.lib, engine.chip
    one = (ctypes.c_int16 * 1)()
    lib.vx_reset(chip)
    lib.vx_inflection(chip, inflection)
    lib.vx_write(chip, phone)
    for n in range(1, limit):
        lib.vx_render(chip, one, 1)
        if lib.vx_ready(chip):
            return n
    return None


def peak(samples):
    return max(abs(v) for v in samples) if samples else 0


def compare_audio(ours, ref, names):
    print('  phone       ours peak   ref peak   max|diff|   ratio')
    rows = []
    for i in range(64):
        a, b = play(ours, [i]), play(ref, [i])
        d = max(abs(x - y) for x, y in zip(a, b))
        pa, pb = peak(a), peak(b)
        rows.append((d, i, names[i], pa, pb))

    exact = sum(1 for r in rows if r[0] == 0)
    within1 = sum(1 for r in rows if r[0] <= 1)
    for d, i, n, pa, pb in sorted(rows, reverse=True)[:10]:
        ratio = ('%.2fx' % (pa / pb)) if pb else '-'
        print('    %02X %-4s  %8d   %8d   %9d   %6s' % (i, n, pa, pb, d, ratio))
    print('    %d of 64 identical, %d within one LSB' % (exact, within1))
    return rows


def compare_timing(ours, ref, names, phones):
    print('  phone       ours    reference   delta')
    deltas = set()
    for i in phones:
        a, b = ready_at(ours, i), ready_at(ref, i)
        deltas.add(None if a is None or b is None else b - a)
        print('    %02X %-4s  %6s   %9s   %+5s'
              % (i, names[i], a, b, 'n/a' if a is None or b is None else b - a))
    if len(deltas) == 1:
        d = deltas.pop()
        print('    constant offset of %+d samples across every phone' % d)


def compare_front_ends(ours, ref, names):
    def hist(call, wide):
        counts = collections.Counter()
        for line in CORPUS:
            if wide:
                buf = (ctypes.c_ubyte * 4096)()
                n = call(line.encode('cp1252', 'replace'), buf, 4096)
                counts.update(buf[i] & 0x3F for i in range(min(n, 4096)))
            else:
                buf = ctypes.create_string_buffer(4096)
                n = call(line.encode('cp1252', 'replace'), buf, 4096)
                counts.update(buf.raw[i] & 0x3F for i in range(min(n, 4096)))
        return counts

    a = hist(ours.lib.ttv_translate, True)
    b = hist(ref.lib.ttv_translate, False)

    only_ref = [i for i in range(64) if not a.get(i) and b.get(i)]
    only_ours = [i for i in range(64) if a.get(i) and not b.get(i)]
    neither = [i for i in range(64) if not a.get(i) and not b.get(i)]

    for i in only_ref:
        print('    OURS NEVER EMITS %-4s -- the reference emits it %d times'
              % (names[i], b[i]))
    for i in only_ours:
        print('    only ours emits  %-4s (%d times)' % (names[i], a[i]))
    if not only_ref and not only_ours:
        print('    both front ends reach the same set of phones')
    print('    neither front end ever emits %d of 64: %s'
          % (len(neither), ' '.join(names[i] for i in neither)))
    print('    (those are the chip\'s duration variants -- reachable only by '
          'writing phones directly)')


def main(argv):
    if not argv:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    dll = argv[0]
    rom_dir = argv[1] if len(argv) > 1 else DEFAULT_ROMS
    if not os.path.isfile(dll):
        print('no such reference library: %s' % dll)
        return 2
    if not os.path.isfile(OURS):
        print('build ours first: python nvda-addon/package.py')
        return 2
    if not all(os.path.isfile(os.path.join(rom_dir, n)) for n in ('sc01.bin', 'sc01a.bin')):
        print('the reference engine needs the mask ROM dumps, which are not in this '
              'repository; pass their folder or set VOTRAX_ROM_DIR '
              '(see reference/roms/README.md). Looked in: %s' % rom_dir)
        return 2

    probe = Ours(0)
    names = [probe.lib.vx_phone_name(i).decode() for i in range(64)]
    probe.close()

    for mask in (0, 1):
        title = 'SC-01-A' if mask == 0 else 'SC-01'
        ours = Ours(mask)
        ref = Reference(dll, rom_dir, mask)
        print('=== %s: audio, identical fixed hold ===' % title)
        compare_audio(ours, ref, names)
        if mask == 0:
            print()
            print('=== phone-end timing, polled one sample at a time ===')
            compare_timing(ours, ref, names,
                           [0x03, 0x18, 0x19, 0x24, 0x2A, 0x3E, 0x3F])
            print()
            print('=== front ends, over %d sentences ===' % len(CORPUS))
            compare_front_ends(ours, ref, names)
        print()
        ours.close()
        ref.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
