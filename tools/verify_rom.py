#!/usr/bin/env python3
"""Check every transcription of the SC-01 mask ROM against the silicon.

The phoneme tables in this repository exist three times: as the die
transcription in `reference/gate-sim/rom.cc`, as C arrays in
`src/votrax_rom.c`, and as Python tuples in `py_emu/rom.py`. Three
transcriptions that agree is the reason the tables can be trusted; agreeing
with the two dumped 512-byte mask ROMs is the reason they can be trusted
against the chip.

The dumps are not in this repository and must not be: they are the contents
of a commercial chip, not anyone's work here, and their copyright is uncleared
(see reference/roms/README.md). Supply your own copies of the standard MAME
files -- in reference/roms/, where .gitignore keeps them out of commits, or in
any folder named by the VOTRAX_ROM_DIR environment variable. Without them this
checks the three transcriptions against each other and says that it did not
check them against the silicon.

It works from the checked-in text rather than from anything that has to be
built or imported, so it runs on a clean tree with no extension compiled:

    python tools/verify_rom.py           # report; exit 1 on any mismatch
    python tools/verify_rom.py -v        # also print the decoded mask delta

The dumps are laid out the way the die is wired, which is worth knowing before
reading any of it: the ROM is content-addressed. Each row is a little-endian
64-bit word carrying its own phone number in bits 56-61, and the rows are not
in phone order -- which is why MAME scans all 64 looking for a match instead of
indexing. Bits 44-55 are unused and read as zero.
"""

from __future__ import annotations

import binascii
import hashlib
import os
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The two production masks, and what their dumps must hash to. These are the
#: standard MAME dumps -- the same CRC32/SHA-1 pairs appear in the ROM_LOAD
#: lines of reference/mame_votrax.cpp.
DUMPS = {
    "sc01a": ("fc416227", "1d6da90b1807a01b5e186ef08476119a862b5e6d"),
    "sc01": ("528d1c57", "268b5884dce04e49e2376df3e2dc82e852b708c1"),
}

#: Where the dumps are looked for: VOTRAX_ROM_DIR if set, else reference/roms/.
ROM_DIR = Path(os.environ["VOTRAX_ROM_DIR"]) if os.environ.get("VOTRAX_ROM_DIR") \
    else ROOT / "reference" / "roms"


def dumps_available() -> bool:
    """True when both dumps have been supplied."""
    return all((ROM_DIR / f"{name}.bin").is_file() for name in DUMPS)

#: The twelve parameters a phone's two words decode to, in slot order.
PARAM_FIELDS = ("f1", "va", "f2", "fc", "f2q", "f3", "fa",
                "cld", "vd", "closure", "duration", "pause")


class VerifyError(AssertionError):
    """A transcription disagrees with the silicon."""


# --- the dumps ------------------------------------------------------------

def load_dump(name: str) -> dict[int, tuple[int, int]]:
    """Decode one 512-byte mask ROM dump into {phone: (word0, word1)}.

    Raises VerifyError if the file is the wrong size, if any row's unused bits
    are set, or if the 64 rows do not carry the 64 distinct phone numbers.
    """
    data = (ROM_DIR / f"{name}.bin").read_bytes()
    if len(data) != 512:
        raise VerifyError(f"{name}.bin is {len(data)} bytes; the SC-01 mask ROM is 512")

    rows: dict[int, tuple[int, int]] = {}
    for i in range(64):
        (val,) = struct.unpack_from("<Q", data, i * 8)
        if (val >> 44) & 0xFFF:
            raise VerifyError(f"{name}.bin row {i}: unused bits 44-55 are set")
        phone = (val >> 56) & 0x3F
        if phone in rows:
            raise VerifyError(f"{name}.bin row {i}: phone {phone:#04x} appears twice")
        rows[phone] = ((val >> 32) & 0xFFF, val & 0xFFFFFFFF)

    if sorted(rows) != list(range(64)):
        raise VerifyError(f"{name}.bin does not carry all 64 phones")
    return rows


def dump_order(name: str) -> list[int]:
    """The phone number of each physical row, in file order."""
    data = (ROM_DIR / f"{name}.bin").read_bytes()
    return [(struct.unpack_from("<Q", data, i * 8)[0] >> 56) & 0x3F for i in range(64)]


def dump_hashes(name: str) -> tuple[str, str]:
    """(crc32, sha1) of a dump, as lowercase hex."""
    data = (ROM_DIR / f"{name}.bin").read_bytes()
    return "%08x" % (binascii.crc32(data) & 0xFFFFFFFF), hashlib.sha1(data).hexdigest()


# --- the transcriptions ---------------------------------------------------

def _block(text: str, opener: str, closer: str) -> str:
    start = text.index(opener) + len(opener)
    return text[start:text.index(closer, start)]


def _hexes(chunk: str) -> list[int]:
    return [int(x, 16) for x in re.findall(r"0[xX]([0-9A-Fa-f]+)", chunk)]


def c_tables() -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    """The SC-01-A rows and the SC-01 word1 deltas from src/votrax_rom.c."""
    text = (ROOT / "src" / "votrax_rom.c").read_text(encoding="utf-8")
    w0 = _hexes(_block(text, "RAW_ROM_W0[64] = {", "};"))
    w1 = _hexes(_block(text, "RAW_ROM_W1[64] = {", "};"))
    if len(w0) != 64 or len(w1) != 64:
        raise VerifyError(f"votrax_rom.c: got {len(w0)} word0 and {len(w1)} word1, want 64 each")
    deltas = _pairs(_block(text, "SC01_W1_DELTAS[12] = {", "};"))
    return {i: (w0[i], w1[i]) for i in range(64)}, deltas


def gate_sim_table() -> dict[int, tuple[int, int]]:
    """The SC-01-A rows from Galibert's die transcription, reference/gate-sim/rom.cc."""
    text = (ROOT / "reference" / "gate-sim" / "rom.cc").read_text(encoding="utf-8")
    rows = re.findall(r"\{\s*0[xX]([0-9A-Fa-f]+)\s*,\s*0[xX]([0-9A-Fa-f]+)\s*\}",
                      _block(text, "rom[64][2] = {", "\n};"))
    if len(rows) != 64:
        raise VerifyError(f"gate-sim/rom.cc: got {len(rows)} rows, want 64")
    return {i: (int(a, 16), int(b, 16)) for i, (a, b) in enumerate(rows)}


def py_emu_tables() -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    """The SC-01-A rows and the SC-01 word1 deltas from py_emu/rom.py."""
    text = (ROOT / "py_emu" / "rom.py").read_text(encoding="utf-8")
    rows = re.findall(r"\(\s*0[xX]([0-9A-Fa-f]+)\s*,\s*0[xX]([0-9A-Fa-f]+)\s*\)",
                      _block(text, "_RAW_ROM = [", "\n]"))
    if len(rows) != 64:
        raise VerifyError(f"py_emu/rom.py: got {len(rows)} rows, want 64")
    deltas = _pairs(_block(text, "_SC01_W1_DELTAS = {", "\n}"), sep=":")
    return {i: (int(a, 16), int(b, 16)) for i, (a, b) in enumerate(rows)}, deltas


def _pairs(chunk: str, sep: str = ",") -> dict[int, int]:
    """{phone: word1} out of a `0x08, 0xC4E9C1A3` or `0x08: 0xC4E9C1A3` list."""
    found = re.findall(r"0[xX]([0-9A-Fa-f]{2})\s*%s\s*0[xX]([0-9A-Fa-f]{8})" % re.escape(sep),
                       chunk)
    return {int(p, 16): int(w, 16) for p, w in found}


# --- decoding, a fourth time ----------------------------------------------
#
# Mirrors reference/gate-sim/rom.cc and src/votrax_rom.c. It is here so that
# the report can say *which field* moved between the masks straight from the
# silicon, rather than repeating what a table in the repository claims.

def _param(word1: int, slot: int) -> int:
    base = word1 >> slot
    return ((8 if base & 0x000001 else 0) |
            (4 if base & 0x000080 else 0) |
            (2 if base & 0x004000 else 0) |
            (1 if base & 0x200000 else 0))


def _clvd(word0: int, word1: int, slot: int) -> int:
    base = (word1 >> 28) | (word0 << 4)
    if slot == 6:
        base >>= 1
    return ((1 if base & 0x01 else 0) |
            (2 if base & 0x04 else 0) |
            (4 if base & 0x10 else 0) |
            (8 if base & 0x40 else 0))


def decode(phone: int, word0: int, word1: int) -> dict[str, int]:
    """The twelve parameters the chip reads for one phone."""
    duration = ((0x40 if word0 & 0x020 else 0) |
                (0x20 if word0 & 0x040 else 0) |
                (0x10 if word0 & 0x080 else 0) |
                (0x08 if word0 & 0x100 else 0) |
                (0x04 if word0 & 0x200 else 0) |
                (0x02 if word0 & 0x400 else 0) |
                (0x01 if word0 & 0x800 else 0)) ^ 0x7F
    return {
        "f1": _param(word1, 0), "va": _param(word1, 1), "f2": _param(word1, 2),
        "fc": _param(word1, 3), "f2q": _param(word1, 4), "f3": _param(word1, 5),
        "fa": _param(word1, 6),
        "cld": _clvd(word0, word1, 0), "vd": _clvd(word0, word1, 6),
        "closure": 1 if word0 & 0x10 else 0,
        "duration": duration,
        "pause": 1 if phone in (0x03, 0x3E) else 0,
    }


def mask_delta() -> dict[int, dict[str, tuple[int, int]]]:
    """{phone: {field: (SC-01-A value, SC-01 value)}} decoded from the dumps."""
    a, o = load_dump("sc01a"), load_dump("sc01")
    out: dict[int, dict[str, tuple[int, int]]] = {}
    for phone in range(64):
        da = decode(phone, *a[phone])
        do = decode(phone, *o[phone])
        moved = {f: (da[f], do[f]) for f in PARAM_FIELDS if da[f] != do[f]}
        if moved:
            out[phone] = moved
    return out


# --- the check ------------------------------------------------------------

def transcriptions_agree() -> list[str]:
    """The three transcriptions against each other, with no dump needed."""
    c_rows, c_deltas = c_tables()
    py_rows, py_deltas = py_emu_tables()
    gate = gate_sim_table()
    for source, rows in (("py_emu/rom.py", py_rows), ("reference/gate-sim/rom.cc", gate)):
        bad = [i for i in range(64) if rows[i] != c_rows[i]]
        if bad:
            raise VerifyError(f"{source} disagrees with src/votrax_rom.c at "
                              f"{len(bad)} phone(s); first is {bad[0]:#04x}")
    if py_deltas != c_deltas:
        raise VerifyError("py_emu/rom.py and src/votrax_rom.c disagree on the SC-01 deltas")
    return ["src/votrax_rom.c, py_emu/rom.py, reference/gate-sim/rom.cc: "
            "64/64 rows agree with each other",
            f"SC-01 delta tables agree: {len(c_deltas)} rows"]


def cross_check() -> list[str]:
    """Run every check. Returns the report lines; raises VerifyError on failure.

    Without the dumps, only the transcriptions are compared with each other."""
    if not dumps_available():
        return transcriptions_agree() + [
            f"dumps not supplied in {ROM_DIR}: NOT checked against the silicon"]
    lines: list[str] = []

    for name, (want_crc, want_sha) in DUMPS.items():
        crc, sha = dump_hashes(name)
        if (crc, sha) != (want_crc, want_sha):
            raise VerifyError(
                f"{name}.bin: CRC32 {crc} SHA-1 {sha}; expected {want_crc} / {want_sha}")
        lines.append(f"{name}.bin  512 bytes  crc32 {crc}  sha1 {sha}")

    silicon = load_dump("sc01a")
    order = dump_order("sc01a")
    if order == list(range(64)):
        raise VerifyError("sc01a.bin rows are in phone order; the die's are not")
    lines.append(f"content-addressed: physical row order starts "
                 f"{', '.join('%02X' % p for p in order[:6])} ...")

    c_rows, c_deltas = c_tables()
    py_rows, py_deltas = py_emu_tables()
    transcriptions = {
        "src/votrax_rom.c": c_rows,
        "py_emu/rom.py": py_rows,
        "reference/gate-sim/rom.cc": gate_sim_table(),
    }
    for source, rows in transcriptions.items():
        bad = [i for i in range(64) if rows[i] != silicon[i]]
        if bad:
            i = bad[0]
            raise VerifyError(
                f"{source} disagrees with sc01a.bin at {len(bad)} phone(s); "
                f"first is {i:#04x}: table has {rows[i][0]:#05x}/{rows[i][1]:#010x}, "
                f"silicon has {silicon[i][0]:#05x}/{silicon[i][1]:#010x}")
        lines.append(f"{source:<28} 64/64 rows match sc01a.bin")

    # The 1980 mask, which the repository stores as a delta rather than a table.
    original = load_dump("sc01")
    same_w0 = [i for i in range(64) if original[i][0] != silicon[i][0]]
    if same_w0:
        raise VerifyError(f"word0 differs between the masks at {same_w0}; it should not")
    silicon_deltas = {i: original[i][1] for i in range(64)
                      if original[i][1] != silicon[i][1]}
    for source, deltas in (("src/votrax_rom.c", c_deltas), ("py_emu/rom.py", py_deltas)):
        if deltas != silicon_deltas:
            raise VerifyError(
                f"{source} SC-01 delta table does not match the two dumps: "
                f"table has {sorted('%02X' % p for p in deltas)}, "
                f"silicon has {sorted('%02X' % p for p in silicon_deltas)}")
        lines.append(f"{source:<28} {len(deltas)}/{len(silicon_deltas)} "
                     f"SC-01 deltas match sc01.bin")

    moved = mask_delta()
    fields = {f for per_phone in moved.values() for f in per_phone}
    if fields != {"va"}:
        raise VerifyError(f"the masks differ in {sorted(fields)}; only va should move")
    if any(sc01 != 15 for per_phone in moved.values() for _, sc01 in per_phone.values()):
        raise VerifyError("the 1980 mask should run all twelve at va=15")
    lines.append(f"mask delta: {len(moved)} phones, va only, "
                 f"{', '.join('%02X' % p for p in sorted(moved))}")

    return lines


def main(argv: list[str]) -> int:
    verbose = "-v" in argv or "--verbose" in argv
    try:
        for line in cross_check():
            print(line)
    except VerifyError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if verbose and dumps_available():
        print()
        print("phone   SC-01-A va   SC-01 va")
        for phone, moved in sorted(mask_delta().items()):
            a, o = moved["va"]
            print(f"  {phone:02X}      {a:>2}          {o:>2}")
    if dumps_available():
        print("\nOK: every transcription agrees with the silicon.")
    else:
        print("\nOK: the transcriptions agree with each other. Supply the dumps "
              "(reference/roms/README.md) to check them against the silicon.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
