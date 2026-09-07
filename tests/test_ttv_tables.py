"""Structural checks on the letter-to-sound tables.

src/ttv_tables.c is data, not code, so a wrong row there is not a compile
error and not a crash — it is one word coming out mispronounced, which nothing
else would catch. These tests read the source and check its shape. The
important one is `test_outputs_are_real_phone_names`: it proves every SC-01
phone the ARPABET map can emit actually exists on the chip.
"""

import re
from pathlib import Path

import pytest

from pyvotrax.phonemes import PHONE_TABLE

TABLES = Path(__file__).resolve().parent.parent / "src" / "ttv_tables.c"


def _quoted_rows(table_name: str, columns: int) -> list[tuple[str, ...]]:
    """Pull the `{ "a", "b", ... }` rows out of one table in the source."""
    text = TABLES.read_text(encoding="utf-8")
    start = text.index(table_name + "[")
    start = text.index("{", start)
    depth, end = 0, start
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    body = text[start + 1:end]
    rows = []
    for row in re.findall(r"\{([^{}]*)\}", body):
        fields = re.findall(r'"((?:[^"]*))"', row)
        if len(fields) == columns:
            rows.append(tuple(fields))
    return rows


@pytest.fixture(scope="module")
def arpa_map():
    return _quoted_rows("TTV_ARPABET", 4)


class TestTableSizes:
    def test_tables_exist(self):
        assert TABLES.is_file()

    def test_rule_group_count(self):
        text = TABLES.read_text(encoding="utf-8")
        assert "const ttv_rule_group TTV_NRL_RULES[27]" in text
        # punctuation plus one group per letter
        names = re.findall(r"static const ttv_rule NRL_RULES_(\w+)\[\]", text)
        assert names == ["PUNCT"] + [chr(ord("A") + i) for i in range(26)]

    def test_total_rule_count(self):
        text = TABLES.read_text(encoding="utf-8")
        total = sum(len(_quoted_rows("NRL_RULES_" + n, 4))
                    for n in ["PUNCT"] + [chr(ord("A") + i) for i in range(26)])
        assert total == 355
        assert "355 rules in 27" in text

    def test_arpabet_map_size(self, arpa_map):
        assert len(arpa_map) == 81

    def test_ascii_and_number_tables(self):
        assert len(_quoted_rows("TTV_EXCEPTIONS", 2)) == 17
        text = TABLES.read_text(encoding="utf-8")
        assert len(re.findall(r'"[^"]*"', text.split("TTV_ASCII_NAMES[128]")[1]
                              .split("};")[0])) == 128


class TestArpabetMap:
    def test_outputs_are_real_phone_names(self, arpa_map):
        """Every phone the map can emit must exist in the chip's 64."""
        for left, arpa, right, out in arpa_map:
            for phone in out.split():
                assert phone in PHONE_TABLE, (arpa, out, phone)

    def test_contexts_are_arpabet_symbols(self, arpa_map):
        symbols = {row[1] for row in arpa_map}
        for left, arpa, right, out in arpa_map:
            for ctx in (left, right):
                assert ctx == "" or ctx in symbols, (arpa, ctx)

    def test_specific_phones_are_covered(self, arpa_map):
        covered = {row[1] for row in arpa_map}
        # the vowels and consonants any English front end has to be able to say
        for arpa in ("IY", "IH", "EH", "AE", "AA", "AO", "OW", "UH", "UW",
                     "ER", "AX", "AH", "AY", "AW", "OY", "EY",
                     "P", "B", "T", "D", "K", "G", "F", "V", "TH", "DH",
                     "S", "Z", "SH", "ZH", "HH", "CH", "JH", "M", "N",
                     "L", "R", "W", "Y"):
            assert arpa in covered, arpa

    def test_context_free_fallback_exists_for_every_symbol(self, arpa_map):
        """Each symbol ends with an unconditional rule, so lookup never fails."""
        last_unconditional = {}
        for left, arpa, right, out in arpa_map:
            if left == "" and right == "":
                last_unconditional[arpa] = out
        assert set(last_unconditional) == {row[1] for row in arpa_map}

    def test_diphthongs_are_spelled_as_glides(self, arpa_map):
        """The SC-01 has no diphthongs; the map builds them from two phones."""
        fallback = {arpa: out for left, arpa, right, out in arpa_map
                    if left == "" and right == ""}
        for arpa in ("EY", "OW", "AY", "AW", "OY", "UW"):
            assert len(fallback[arpa].split()) >= 2, (arpa, fallback[arpa])
