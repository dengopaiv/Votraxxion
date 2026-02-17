"""Votrax SC-01A ROM data and parameter extraction.

Raw ROM data from rom.cc, with bit extraction matching the original logic.
"""

from typing import NamedTuple

# Raw ROM data: rom[64][2] from rom.cc
_RAW_ROM = [
    (0x361, 0x74688127),
    (0x161, 0xD4688127),
    (0x9A1, 0xC4688127),
    (0x0E0, 0xF0A050A4),
    (0x0FB, 0x610316E8),
    (0x161, 0x64C9C1A6),
    (0x7A1, 0x34C9C1A6),
    (0x463, 0xF3CB546C),
    (0x161, 0xC4E940A3),
    (0xB61, 0x806191A6),
    (0xA61, 0x906191A6),
    (0x9A1, 0x906191A6),
    (0x7A3, 0x66A58832),
    (0xA61, 0xE6241936),
    (0x173, 0x90E19122),
    (0x163, 0xF7D36428),
    (0x163, 0xFB8B546C),
    (0x9A2, 0xFB8B546C),
    (0x163, 0x9CD15860),
    (0x8A0, 0x706980A3),
    (0x9A0, 0xD4084B36),
    (0x8A1, 0x84E940A3),
    (0x7A1, 0x30498123),
    (0xA21, 0x20498123),
    (0x7A1, 0xF409D0A2),
    (0xA72, 0x1123642C),
    (0x0E8, 0xDB7B342C),
    (0x162, 0xFD2204AC),
    (0x173, 0xE041C126),
    (0x7A2, 0x65832CA8),
    (0xB7C, 0x00E89126),
    (0x468, 0x489132E0),
    (0xA21, 0x84C9C1A6),
    (0x561, 0x7069D326),
    (0xA61, 0x64A01226),
    (0x0E3, 0x548981A3),
    (0xCC1, 0x84E940A3),
    (0x7B2, 0x631324A8),
    (0xA21, 0x84E8C1A2),
    (0xA21, 0x806191A6),
    (0xA21, 0x80E8C122),
    (0x7A1, 0x64015326),
    (0x172, 0xE81132E0),
    (0x463, 0x54084382),
    (0xA20, 0x7049D326),
    (0xA66, 0x1460C122),
    (0xA20, 0x74E880A7),
    (0x7A0, 0x74E880A7),
    (0x461, 0x606980A3),
    (0x163, 0x548981A3),
    (0x7A1, 0xE48981A3),
    (0xA21, 0xB48981A3),
    (0xA61, 0x34E8C1A2),
    (0x9A1, 0x80E8C1A2),
    (0x366, 0x106083A2),
    (0x461, 0x90E8C122),
    (0xA63, 0x88E15220),
    (0x168, 0x183800A4),
    (0x8A1, 0x2448C382),
    (0xA21, 0x94688127),
    (0x9A1, 0x9049D326),
    (0xCC1, 0xB06980A3),
    (0xA23, 0x00A050A4),
    (0x0F0, 0x30A058A4),
]


class PhonemeParams(NamedTuple):
    """Decoded ROM parameters for a single phoneme."""
    f1: int        # 4-bit filter 1 frequency
    va: int        # 4-bit voice amplitude
    f2: int        # 4-bit filter 2 frequency (treated as 5-bit in filter commit)
    fc: int        # 4-bit fricative/noise control
    f2q: int       # 4-bit filter 2 Q factor
    f3: int        # 4-bit filter 3 frequency
    fa: int        # 4-bit fricative amplitude
    cld: int       # 4-bit closure delay
    vd: int        # 4-bit voice delay
    closure: int   # 1-bit closure flag
    duration: int  # 7-bit duration
    pause: bool    # True if this is a pause phoneme


def _extract_param(word1: int, slot: int) -> int:
    """Extract a 4-bit parameter from a ROM slot (0-6).

    Matches rom.cc bit extraction:
    bit0 = (base >> 0) & 1, bit1 = (base >> 7) & 1,
    bit2 = (base >> 14) & 1, bit3 = (base >> 21) & 1
    """
    base = word1 >> slot
    return (
        (8 if (base & 0x000001) else 0) |
        (4 if (base & 0x000080) else 0) |
        (2 if (base & 0x004000) else 0) |
        (1 if (base & 0x200000) else 0)
    )


def _extract_clvd(word0: int, word1: int, slot: int) -> int:
    """Extract 4-bit cld or vd value.

    slot=0 for cld (f1 slot), slot=6 for vd (fa slot).
    """
    base = (word1 >> 28) | (word0 << 4)
    if slot == 6:
        base >>= 1
    return (
        (1 if (base & 0x01) else 0) |
        (2 if (base & 0x04) else 0) |
        (4 if (base & 0x10) else 0) |
        (8 if (base & 0x40) else 0)
    )


def _decode_phoneme(index: int) -> PhonemeParams:
    """Decode ROM data for a single phoneme."""
    word0, word1 = _RAW_ROM[index]

    f1 = _extract_param(word1, 0)
    va = _extract_param(word1, 1)
    f2 = _extract_param(word1, 2)
    fc = _extract_param(word1, 3)
    f2q = _extract_param(word1, 4)
    f3 = _extract_param(word1, 5)
    fa = _extract_param(word1, 6)

    cld = _extract_clvd(word0, word1, 0)
    vd = _extract_clvd(word0, word1, 6)

    closure = 1 if (word0 & 0x10) else 0

    duration = (
        (0x40 if (word0 & 0x020) else 0) |
        (0x20 if (word0 & 0x040) else 0) |
        (0x10 if (word0 & 0x080) else 0) |
        (0x08 if (word0 & 0x100) else 0) |
        (0x04 if (word0 & 0x200) else 0) |
        (0x02 if (word0 & 0x400) else 0) |
        (0x01 if (word0 & 0x800) else 0)
    )

    pause = not (index == 0x03 or index == 0x3E)

    return PhonemeParams(
        f1=f1, va=va, f2=f2, fc=fc, f2q=f2q, f3=f3, fa=fa,
        cld=cld, vd=vd, closure=closure, duration=duration, pause=pause,
    )


# Module-level ROM data, decoded at import time
ROM_DATA = [_decode_phoneme(i) for i in range(64)]
