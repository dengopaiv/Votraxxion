"""Votrax SC-01A phoneme table and lookup utilities."""

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
