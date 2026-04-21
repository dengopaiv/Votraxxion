"""Votrax SC-01A phoneme table and lookup utilities."""

import re

# 64-entry phoneme table from MAME votrax.cpp
PHONE_TABLE = [
    "EH3", "EH2", "EH1", "PA0", "DT",  "A2",  "A1",  "ZH",
    "AH2", "I3",  "I2",  "I1",  "M",   "N",   "B",   "V",
    "CH",  "SH",  "Z",   "AW1", "NG",  "AH1", "OO1", "OO",
    "L",   "K",   "J",   "H",   "G",   "F",   "D",   "S",
    "A",   "AY",  "Y1",  "UH3", "AH",  "P",   "O",   "I",
    "U",   "Y",   "T",   "R",   "E",   "W",   "AE",  "AE1",
    "AW2", "UH2", "UH1", "UH",  "O2",  "O1",  "IU",  "U1",
    "THV", "TH",  "ER",  "EH",  "E1",  "AW",  "PA1", "STOP",
]


def name_to_code(name: str) -> int:
    """Convert a phoneme name to its 6-bit code (0-63).

    Raises ValueError if the name is not found.
    """
    name_upper = name.upper()
    try:
        return PHONE_TABLE.index(name_upper)
    except ValueError:
        raise ValueError(f"Unknown phoneme name: {name!r}")


def code_to_name(code: int) -> str:
    """Convert a 6-bit phoneme code (0-63) to its name.

    Raises IndexError if the code is out of range.
    """
    if not 0 <= code <= 63:
        raise IndexError(f"Phoneme code out of range: {code}")
    return PHONE_TABLE[code]


# Token syntax: NAME[:INFLECTION][*REPEAT]
#   NAME       — phoneme name (case-insensitive, one of PHONE_TABLE)
#   INFLECTION — integer 0–3 (default 0)
#   REPEAT     — positive integer (default 1), commits the phoneme N times
_TOKEN_RE = re.compile(
    r"^([A-Za-z]+[0-9]?)(?::([0-3]))?(?:\*([1-9][0-9]*))?$"
)


def parse_phoneme_sequence(seq: str) -> list[tuple[int, int]]:
    """Parse a phoneme-sequence string into (code, inflection) tuples.

    Input is whitespace-separated tokens of the form ``NAME[:INFLECTION][*REPEAT]``.
    Lines starting with ``#`` and blank lines are ignored so phoneme files can
    carry comments.

    Examples::

        parse_phoneme_sequence("AH")              # [(0x24, 0)]
        parse_phoneme_sequence("AH:2")            # [(0x24, 2)]
        parse_phoneme_sequence("AH*3")            # [(0x24, 0)] * 3
        parse_phoneme_sequence("I3 M P O1:2 R")   # 5 tuples

    Raises ValueError on an unparseable token or unknown phoneme name.
    """
    result: list[tuple[int, int]] = []
    for raw_line in seq.splitlines():
        line = raw_line.split("#", 1)[0]
        for token in line.split():
            m = _TOKEN_RE.match(token)
            if not m:
                raise ValueError(f"Unparseable phoneme token: {token!r}")
            name, infl_s, rep_s = m.groups()
            code = name_to_code(name)
            inflection = int(infl_s) if infl_s is not None else 0
            repeat = int(rep_s) if rep_s is not None else 1
            for _ in range(repeat):
                result.append((code, inflection))
    return result
