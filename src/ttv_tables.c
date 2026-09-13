/** Letter-to-sound tables for a self-contained SC-01 text-to-speech front end.
 *
 * Two stages, both pure data:
 *
 *   1. NRL_RULES -- English spelling to ARPABET, by the Naval Research
 *      Laboratory letter-to-sound ruleset (Elovitz et al., NRL Report 7948,
 *      1976), in the arrangement popularised by John A. Wasser's public-domain
 *      english.c (1985) and shipped in Votrax-era products.  355 rules in 27
 *      groups: punctuation, then one group per letter A-Z.
 *
 *   2. ARPABET_TO_SC01 -- ARPABET to SC-01 phone names, context sensitive:
 *      NRL Report 7948's own "IPA to Votrax translation rules", as transcribed
 *      by Tamas Geczy.  This is the interesting half: it is tuned to the
 *      SC-01's actual phone inventory rather than to a general phoneme set.
 *      Diphthongs are spelled out as two-phone glides (AY -> "AH E1"), affricates as stop+fricative
 *      (CH -> "T CH"), and vowels take an UH3 onglide or an I3/EH3 offglide
 *      next to liquids, which is how the chip's interpolator is coaxed into
 *      sounding like coarticulation.
 *
 * Rule format, matching the NRL convention:
 *
 *   {left context, match, right context, output}
 *
 * The matcher walks the text left to right.  At each position it takes the
 * group for the current character and tries each rule in order; the first
 * whose `match` is present at the cursor and whose contexts both hold wins,
 * its `out` is appended, and the cursor advances by strlen(match).
 *
 * Context character classes:
 *
 *   #  one or more vowels        :  zero or more consonants
 *   ^  one consonant             +  a front vowel (E, I or Y)
 *   %  a suffix -- E, ER, ES, ED, ING or ELY (right context only)
 *   .  a voiced consonant (B D V G J L M N R W Z)
 *
 * Any other character matches itself literally; a space matches a word
 * boundary.  Left contexts are written in reverse reading order, so the
 * character nearest the cursor comes last.
 *
 * Provenance: recovered from the compiled tables of sc01.dll in Tamas Geczy's
 * votraxsc01 NVDA add-on (2026, github.com/tgeczy/votraxsc01-nvda), and since
 * checked entry for entry against his source.  Whose each table is:
 *
 *   NRL_RULES, CARDINALS, ORDINALS, ASCII_NAMES, ABBREVIATIONS
 *       The NRL rules (a US Government work) and Wasser's english.c,
 *       saynum.c, spellword.c and parse.c (1985, public domain), which Geczy
 *       vendored byte-identical.  Seven ASCII_NAMES letters are corrected
 *       here; three of those corrections are his.
 *   ARPABET
 *       Geczy's transcription of the NRL IPA-to-Votrax rules from the
 *       report's SNOBOL listing (his arpabet_to_sc01.c), including his two
 *       documented repairs of typos the surviving transcriptions share: the
 *       unclosed bracket in L EY, and ER before L read as context rather than
 *       consumed.  The rules are public domain; the transcription is
 *       BSD-3-Clause, copyright tgeczy.  81 entries, identical and in order.
 *   EXCEPTIONS
 *       Geczy's exception dictionary (his exceptions.c): respellings he
 *       measured broken through the rules and measured correct after.
 *       BSD-3-Clause, copyright tgeczy.  17 entries, identical.
 *
 * See NOTICE.md.  Kept here as source so the synthesizer needs no dictionary,
 * no data file and no runtime download -- see docs/tech-overview.md, Part 4.
 */

#include "ttv_tables.h"


/* NRL Report 7948 maps every punctuation mark to a space, because it was a
 * letter-to-sound algorithm and had no opinions about timing.  Here that is a
 * bug: the ARPABET map below has entries turning "," into PA1 and "." into
 * PA1 PA1, and a space into PA0, so mapping punctuation to a space made those
 * rows unreachable and every pause in the language came out as the 49 ms PA0
 * instead of the 186 ms PA1.  Commas and full stops were audible only as a
 * catch of breath, which is what made connected speech run together.
 *
 * So the marks are passed through to stage two, which is where this codebase
 * decides what a pause is worth.  "?" and "!" become "." because they end a
 * sentence and should pause like one; their *pitch* comes from the contour,
 * which reads the terminator from the original text and never saw this table.
 *
 * The hyphen still emits nothing, deliberately.  There is a "-" -> PA1 row in
 * the map below, but a hyphen is far more often inside a word ("well-known")
 * than standing alone as a dash, and a 186 ms gap in the middle of a compound
 * is worse than no pause at all. */
static const ttv_rule NRL_RULES_PUNCT[] = {
    { "",        " ",         "",        " " },
    { "",        "-",         "",        "" },
    { ".",       "'S",        "",        "z" },
    { "#:.E",    "'S",        "",        "z" },
    { "#",       "'S",        "",        "z" },
    { "",        "'",         "",        "" },
    { "",        ",",         "",        "," },
    { "",        ".",         "",        "." },
    { "",        "?",         "",        "." },
    { "",        "!",         "",        "." },
};

static const ttv_rule NRL_RULES_A[] = {
    { "",        "A",         " ",       "AX" },
    { " ",       "ARE",       " ",       "AAr" },
    { " ",       "AR",        "O",       "AXr" },
    { "",        "AR",        "#",       "EHr" },
    { "^",       "AS",        "#",       "EYs" },
    { "",        "A",         "WA",      "AX" },
    { "",        "AW",        "",        "AO" },
    { " :",      "ANY",       "",        "EHnIY" },
    { "",        "A",         "^+#",     "EY" },
    { "#:",      "ALLY",      "",        "AXlIY" },
    { " ",       "AL",        "#",       "AXl" },
    { "",        "AGAIN",     "",        "AXgEHn" },
    { "#:",      "AG",        "E",       "IHj" },
    { "",        "A",         "^+:#",    "AE" },
    { " :",      "A",         "^+ ",     "EY" },
    { "",        "A",         "^%",      "EY" },
    { " ",       "ARR",       "",        "AXr" },
    { "",        "ARR",       "",        "AEr" },
    { " :",      "AR",        " ",       "AAr" },
    { "",        "AR",        " ",       "ER" },
    { "",        "AR",        "",        "AAr" },
    { "",        "AIR",       "",        "EHr" },
    { "",        "AI",        "",        "EY" },
    { "",        "AY",        "",        "EY" },
    { "",        "AU",        "",        "AO" },
    { "#:",      "AL",        " ",       "AXl" },
    { "#:",      "ALS",       " ",       "AXlz" },
    { "",        "ALK",       "",        "AOk" },
    { "",        "AL",        "^",       "AOl" },
    { " :",      "ABLE",      "",        "EYbAXl" },
    { "",        "ABLE",      "",        "AXbAXl" },
    { "",        "ANG",       "+",       "EYnj" },
    { "",        "A",         "",        "AE" },
};

static const ttv_rule NRL_RULES_B[] = {
    { " ",       "BE",        "^#",      "bIH" },
    { "",        "BEING",     "",        "bIYIHNG" },
    { " ",       "BOTH",      " ",       "bOWTH" },
    { " ",       "BUS",       "#",       "bIHz" },
    { "",        "BUIL",      "",        "bIHl" },
    { "",        "B",         "",        "b" },
};

static const ttv_rule NRL_RULES_C[] = {
    { " ",       "CH",        "^",       "k" },
    { "^E",      "CH",        "",        "k" },
    { "",        "CH",        "",        "CH" },
    { " S",      "CI",        "#",       "sAY" },
    { "",        "CI",        "A",       "SH" },
    { "",        "CI",        "O",       "SH" },
    { "",        "CI",        "EN",      "SH" },
    { "",        "C",         "+",       "s" },
    { "",        "CK",        "",        "k" },
    { "",        "COM",       "%",       "kAHm" },
    { "",        "C",         "",        "k" },
};

static const ttv_rule NRL_RULES_D[] = {
    { "#:",      "DED",       " ",       "dIHd" },
    { ".E",      "D",         " ",       "d" },
    { "#:^E",    "D",         " ",       "t" },
    { " ",       "DE",        "^#",      "dIH" },
    { " ",       "DO",        " ",       "dUW" },
    { " ",       "DOES",      "",        "dAHz" },
    { " ",       "DOING",     "",        "dUWIHNG" },
    { " ",       "DOW",       "",        "dAW" },
    { "",        "DU",        "A",       "jUW" },
    { "",        "D",         "",        "d" },
};

static const ttv_rule NRL_RULES_E[] = {
    { "#:",      "E",         " ",       "" },
    { "':^",     "E",         " ",       "" },
    { " :",      "E",         " ",       "IY" },
    { "#",       "ED",        " ",       "d" },
    { "#:",      "E",         "D ",      "" },
    { "",        "EV",        "ER",      "EHv" },
    { "",        "E",         "^%",      "IY" },
    { "",        "ERI",       "#",       "IYrIY" },
    { "",        "ERI",       "",        "EHrIH" },
    { "#:",      "ER",        "#",       "ER" },
    { "",        "ER",        "#",       "EHr" },
    { "",        "ER",        "",        "ER" },
    { " ",       "EVEN",      "",        "IYvEHn" },
    { "#:",      "E",         "W",       "" },
    { "T",       "EW",        "",        "UW" },
    { "S",       "EW",        "",        "UW" },
    { "R",       "EW",        "",        "UW" },
    { "D",       "EW",        "",        "UW" },
    { "L",       "EW",        "",        "UW" },
    { "Z",       "EW",        "",        "UW" },
    { "N",       "EW",        "",        "UW" },
    { "J",       "EW",        "",        "UW" },
    { "TH",      "EW",        "",        "UW" },
    { "CH",      "EW",        "",        "UW" },
    { "SH",      "EW",        "",        "UW" },
    { "",        "EW",        "",        "yUW" },
    { "",        "E",         "O",       "IY" },
    { "#:S",     "ES",        " ",       "IHz" },
    { "#:C",     "ES",        " ",       "IHz" },
    { "#:G",     "ES",        " ",       "IHz" },
    { "#:Z",     "ES",        " ",       "IHz" },
    { "#:X",     "ES",        " ",       "IHz" },
    { "#:J",     "ES",        " ",       "IHz" },
    { "#:CH",    "ES",        " ",       "IHz" },
    { "#:SH",    "ES",        " ",       "IHz" },
    { "#:",      "E",         "S ",      "" },
    { "#:",      "ELY",       " ",       "lIY" },
    { "#:",      "EMENT",     "",        "mEHnt" },
    { "",        "EFUL",      "",        "fUHl" },
    { "",        "EE",        "",        "IY" },
    { "",        "EARN",      "",        "ERn" },
    { " ",       "EAR",       "^",       "ER" },
    { "",        "EAD",       "",        "EHd" },
    { "#:",      "EA",        " ",       "IYAX" },
    { "",        "EA",        "SU",      "EH" },
    { "",        "EA",        "",        "IY" },
    { "",        "EIGH",      "",        "EY" },
    { "",        "EI",        "",        "IY" },
    { " ",       "EYE",       "",        "AY" },
    { "",        "EY",        "",        "IY" },
    { "",        "EU",        "",        "yUW" },
    { "",        "E",         "",        "EH" },
};

static const ttv_rule NRL_RULES_F[] = {
    { "",        "FUL",       "",        "fUHl" },
    { "",        "F",         "",        "f" },
};

static const ttv_rule NRL_RULES_G[] = {
    { "",        "GIV",       "",        "gIHv" },
    { " ",       "G",         "I^",      "g" },
    { "",        "GE",        "T",       "gEH" },
    { "SU",      "GGES",      "",        "gjEHs" },
    { "",        "GG",        "",        "g" },
    { " B#",     "G",         "",        "g" },
    { "",        "G",         "+",       "j" },
    { "",        "GREAT",     "",        "grEYt" },
    { "#",       "GH",        "",        "" },
    { "",        "G",         "",        "g" },
};

static const ttv_rule NRL_RULES_H[] = {
    { " ",       "HAV",       "",        "hAEv" },
    { " ",       "HERE",      "",        "hIYr" },
    { " ",       "HOUR",      "",        "AWER" },
    { "",        "HOW",       "",        "hAW" },
    { "",        "H",         "#",       "h" },
    { "",        "H",         "",        "" },
};

static const ttv_rule NRL_RULES_I[] = {
    { " ",       "IN",        "",        "IHn" },
    { " ",       "I",         " ",       "AY" },
    { "",        "IN",        "D",       "AYn" },
    { "",        "IER",       "",        "IYER" },
    { "#:R",     "IED",       "",        "IYd" },
    { "",        "IED",       " ",       "AYd" },
    { "",        "IEN",       "",        "IYEHn" },
    { "",        "IE",        "T",       "AYEH" },
    { " :",      "I",         "%",       "AY" },
    { "",        "I",         "%",       "IY" },
    { "",        "IE",        "",        "IY" },
    { "",        "I",         "^+:#",    "IH" },
    { "",        "IR",        "#",       "AYr" },
    { "",        "IZ",        "%",       "AYz" },
    { "",        "IS",        "%",       "AYz" },
    { "",        "I",         "D%",      "AY" },
    { "+^",      "I",         "^+",      "IH" },
    { "",        "I",         "T%",      "AY" },
    { "#:^",     "I",         "^+",      "IH" },
    { "",        "I",         "^+",      "AY" },
    { "",        "IR",        "",        "ER" },
    { "",        "IGH",       "",        "AY" },
    { "",        "ILD",       "",        "AYld" },
    { "",        "IGN",       " ",       "AYn" },
    { "",        "IGN",       "^",       "AYn" },
    { "",        "IGN",       "%",       "AYn" },
    { "",        "IQUE",      "",        "IYk" },
    { "",        "I",         "",        "IH" },
};

static const ttv_rule NRL_RULES_J[] = {
    { "",        "J",         "",        "j" },
};

static const ttv_rule NRL_RULES_K[] = {
    { " ",       "K",         "N",       "" },
    { "",        "K",         "",        "k" },
};

static const ttv_rule NRL_RULES_L[] = {
    { "",        "LO",        "C#",      "lOW" },
    { "L",       "L",         "",        "" },
    { "#:^",     "L",         "%",       "AXl" },
    { "",        "LEAD",      "",        "lIYd" },
    { "",        "L",         "",        "l" },
};

static const ttv_rule NRL_RULES_M[] = {
    { "",        "MOV",       "",        "mUWv" },
    { "",        "M",         "",        "m" },
};

static const ttv_rule NRL_RULES_N[] = {
    { "E",       "NG",        "+",       "nj" },
    { "",        "NG",        "R",       "NGg" },
    { "",        "NG",        "#",       "NGg" },
    { "",        "NGL",       "%",       "NGgAXl" },
    { "",        "NG",        "",        "NG" },
    { "",        "NK",        "",        "NGk" },
    { " ",       "NOW",       " ",       "nAW" },
    { "",        "N",         "",        "n" },
};

static const ttv_rule NRL_RULES_O[] = {
    { "",        "OF",        " ",       "AXv" },
    { "",        "OROUGH",    "",        "EROW" },
    { "#:",      "OR",        " ",       "ER" },
    { "#:",      "ORS",       " ",       "ERz" },
    { "",        "OR",        "",        "AOr" },
    { " ",       "ONE",       "",        "wAHn" },
    { "",        "OW",        "",        "OW" },
    { " ",       "OVER",      "",        "OWvER" },
    { "",        "OV",        "",        "AHv" },
    { "",        "O",         "^%",      "OW" },
    { "",        "O",         "^EN",     "OW" },
    { "",        "O",         "^I#",     "OW" },
    { "",        "OL",        "D",       "OWl" },
    { "",        "OUGHT",     "",        "AOt" },
    { "",        "OUGH",      "",        "AHf" },
    { " ",       "OU",        "",        "AW" },
    { "H",       "OU",        "S#",      "AW" },
    { "",        "OUS",       "",        "AXs" },
    { "",        "OUR",       "",        "AOr" },
    { "",        "OULD",      "",        "UHd" },
    { "^",       "OU",        "^L",      "AH" },
    { "",        "OUP",       "",        "UWp" },
    { "",        "OU",        "",        "AW" },
    { "",        "OY",        "",        "OY" },
    { "",        "OING",      "",        "OWIHNG" },
    { "",        "OI",        "",        "OY" },
    { "",        "OOR",       "",        "AOr" },
    { "",        "OOK",       "",        "UHk" },
    { "",        "OOD",       "",        "UHd" },
    { "",        "OO",        "",        "UW" },
    { "",        "O",         "E",       "OW" },
    { "",        "O",         " ",       "OW" },
    { "",        "OA",        "",        "OW" },
    { " ",       "ONLY",      "",        "OWnlIY" },
    { " ",       "ONCE",      "",        "wAHns" },
    { "",        "ON'T",      "",        "OWnt" },
    { "C",       "O",         "N",       "AA" },
    { "",        "O",         "NG",      "AO" },
    { " :^",     "O",         "N",       "AH" },
    { "I",       "ON",        "",        "AXn" },
    { "#:",      "ON",        " ",       "AXn" },
    { "#^",      "ON",        "",        "AXn" },
    { "",        "O",         "ST ",     "OW" },
    { "",        "OF",        "^",       "AOf" },
    { "",        "OTHER",     "",        "AHDHER" },
    { "",        "OSS",       " ",       "AOs" },
    { "#:^",     "OM",        "",        "AHm" },
    { "",        "O",         "",        "AA" },
};

static const ttv_rule NRL_RULES_P[] = {
    { "",        "PH",        "",        "f" },
    { "",        "PEOP",      "",        "pIYp" },
    { "",        "POW",       "",        "pAW" },
    { "",        "PUT",       " ",       "pUHt" },
    { "",        "P",         "",        "p" },
};

static const ttv_rule NRL_RULES_Q[] = {
    { "",        "QUAR",      "",        "kwAOr" },
    { "",        "QU",        "",        "kw" },
    { "",        "Q",         "",        "k" },
};

static const ttv_rule NRL_RULES_R[] = {
    { " ",       "RE",        "^#",      "rIY" },
    { "",        "R",         "",        "r" },
};

static const ttv_rule NRL_RULES_S[] = {
    { "",        "SH",        "",        "SH" },
    { "#",       "SION",      "",        "ZHAXn" },
    { "",        "SOME",      "",        "sAHm" },
    { "#",       "SUR",       "#",       "ZHER" },
    { "",        "SUR",       "#",       "SHER" },
    { "#",       "SU",        "#",       "ZHUW" },
    { "#",       "SSU",       "#",       "SHUW" },
    { "#",       "SED",       " ",       "zd" },
    { "#",       "S",         "#",       "z" },
    { "",        "SAID",      "",        "sEHd" },
    { "^",       "SION",      "",        "SHAXn" },
    { "",        "S",         "S",       "" },
    { ".",       "S",         " ",       "z" },
    { "#:.E",    "S",         " ",       "z" },
    { "#:^##",   "S",         " ",       "z" },
    { "#:^#",    "S",         " ",       "s" },
    { "U",       "S",         " ",       "s" },
    { " :#",     "S",         " ",       "z" },
    { " ",       "SCH",       "",        "sk" },
    { "",        "S",         "C+",      "" },
    { "#",       "SM",        "",        "zm" },
    { "#",       "SN",        "'",       "zAXn" },
    { "",        "S",         "",        "s" },
};

static const ttv_rule NRL_RULES_T[] = {
    { " ",       "THE",       " ",       "DHAX" },
    { "",        "TO",        " ",       "tUW" },
    { "",        "THAT",      " ",       "DHAEt" },
    { " ",       "THIS",      " ",       "DHIHs" },
    { " ",       "THEY",      "",        "DHEY" },
    { " ",       "THERE",     "",        "DHEHr" },
    { "",        "THER",      "",        "DHER" },
    { "",        "THEIR",     "",        "DHEHr" },
    { " ",       "THAN",      " ",       "DHAEn" },
    { " ",       "THEM",      " ",       "DHEHm" },
    { "",        "THESE",     " ",       "DHIYz" },
    { " ",       "THEN",      "",        "DHEHn" },
    { "",        "THROUGH",   "",        "THrUW" },
    { "",        "THOSE",     "",        "DHOWz" },
    { "",        "THOUGH",    " ",       "DHOW" },
    { " ",       "THUS",      "",        "DHAHs" },
    { "",        "TH",        "",        "TH" },
    { "#:",      "TED",       " ",       "tIHd" },
    { "S",       "TI",        "#N",      "CH" },
    { "",        "TI",        "O",       "SH" },
    { "",        "TI",        "A",       "SH" },
    { "",        "TIEN",      "",        "SHAXn" },
    { "",        "TUR",       "#",       "CHER" },
    { "",        "TU",        "A",       "CHUW" },
    { " ",       "TWO",       "",        "tUW" },
    { "",        "T",         "",        "t" },
};

static const ttv_rule NRL_RULES_U[] = {
    { " ",       "UN",        "I",       "yUWn" },
    { " ",       "UN",        "",        "AHn" },
    { " ",       "UPON",      "",        "AXpAOn" },
    { "T",       "UR",        "#",       "UHr" },
    { "S",       "UR",        "#",       "UHr" },
    { "R",       "UR",        "#",       "UHr" },
    { "D",       "UR",        "#",       "UHr" },
    { "L",       "UR",        "#",       "UHr" },
    { "Z",       "UR",        "#",       "UHr" },
    { "N",       "UR",        "#",       "UHr" },
    { "J",       "UR",        "#",       "UHr" },
    { "TH",      "UR",        "#",       "UHr" },
    { "CH",      "UR",        "#",       "UHr" },
    { "SH",      "UR",        "#",       "UHr" },
    { "",        "UR",        "#",       "yUHr" },
    { "",        "UR",        "",        "ER" },
    { "",        "U",         "^ ",      "AH" },
    { "",        "U",         "^^",      "AH" },
    { "",        "UY",        "",        "AY" },
    { " G",      "U",         "#",       "" },
    { "G",       "U",         "%",       "" },
    { "G",       "U",         "#",       "w" },
    { "#N",      "U",         "",        "yUW" },
    { "T",       "U",         "",        "UW" },
    { "S",       "U",         "",        "UW" },
    { "R",       "U",         "",        "UW" },
    { "D",       "U",         "",        "UW" },
    { "L",       "U",         "",        "UW" },
    { "Z",       "U",         "",        "UW" },
    { "N",       "U",         "",        "UW" },
    { "J",       "U",         "",        "UW" },
    { "TH",      "U",         "",        "UW" },
    { "CH",      "U",         "",        "UW" },
    { "SH",      "U",         "",        "UW" },
    { "",        "U",         "",        "yUW" },
};

static const ttv_rule NRL_RULES_V[] = {
    { "",        "VIEW",      "",        "vyUW" },
    { "",        "V",         "",        "v" },
};

static const ttv_rule NRL_RULES_W[] = {
    { " ",       "WERE",      "",        "wER" },
    { "",        "WA",        "S",       "wAA" },
    { "",        "WA",        "T",       "wAA" },
    { "",        "WHERE",     "",        "WHEHr" },
    { "",        "WHAT",      "",        "WHAAt" },
    { "",        "WHOL",      "",        "hOWl" },
    { "",        "WHO",       "",        "hUW" },
    { "",        "WH",        "",        "WH" },
    { "",        "WAR",       "",        "wAOr" },
    { "",        "WOR",       "^",       "wER" },
    { "",        "WR",        "",        "r" },
    { "",        "W",         "",        "w" },
};

static const ttv_rule NRL_RULES_X[] = {
    { "",        "X",         "",        "ks" },
};

static const ttv_rule NRL_RULES_Y[] = {
    { "",        "YOUNG",     "",        "yAHNG" },
    { " ",       "YOU",       "",        "yUW" },
    { " ",       "YES",       "",        "yEHs" },
    { " ",       "Y",         "",        "y" },
    { "#:^",     "Y",         " ",       "IY" },
    { "#:^",     "Y",         "I",       "IY" },
    { " :",      "Y",         " ",       "AY" },
    { " :",      "Y",         "#",       "AY" },
    { " :",      "Y",         "^+:#",    "IH" },
    { " :",      "Y",         "^#",      "AY" },
    { "",        "Y",         "",        "IH" },
};

static const ttv_rule NRL_RULES_Z[] = {
    { "",        "Z",         "",        "z" },
};

// Indexed by letter: [0] is the punctuation/space group, [1 + c - 'A'] the
// group for an upper-case letter.

const ttv_rule_group TTV_NRL_RULES[27] = {
    { NRL_RULES_PUNCT,  sizeof(NRL_RULES_PUNCT) / sizeof(ttv_rule) },
    { NRL_RULES_A,      sizeof(NRL_RULES_A) / sizeof(ttv_rule) },
    { NRL_RULES_B,      sizeof(NRL_RULES_B) / sizeof(ttv_rule) },
    { NRL_RULES_C,      sizeof(NRL_RULES_C) / sizeof(ttv_rule) },
    { NRL_RULES_D,      sizeof(NRL_RULES_D) / sizeof(ttv_rule) },
    { NRL_RULES_E,      sizeof(NRL_RULES_E) / sizeof(ttv_rule) },
    { NRL_RULES_F,      sizeof(NRL_RULES_F) / sizeof(ttv_rule) },
    { NRL_RULES_G,      sizeof(NRL_RULES_G) / sizeof(ttv_rule) },
    { NRL_RULES_H,      sizeof(NRL_RULES_H) / sizeof(ttv_rule) },
    { NRL_RULES_I,      sizeof(NRL_RULES_I) / sizeof(ttv_rule) },
    { NRL_RULES_J,      sizeof(NRL_RULES_J) / sizeof(ttv_rule) },
    { NRL_RULES_K,      sizeof(NRL_RULES_K) / sizeof(ttv_rule) },
    { NRL_RULES_L,      sizeof(NRL_RULES_L) / sizeof(ttv_rule) },
    { NRL_RULES_M,      sizeof(NRL_RULES_M) / sizeof(ttv_rule) },
    { NRL_RULES_N,      sizeof(NRL_RULES_N) / sizeof(ttv_rule) },
    { NRL_RULES_O,      sizeof(NRL_RULES_O) / sizeof(ttv_rule) },
    { NRL_RULES_P,      sizeof(NRL_RULES_P) / sizeof(ttv_rule) },
    { NRL_RULES_Q,      sizeof(NRL_RULES_Q) / sizeof(ttv_rule) },
    { NRL_RULES_R,      sizeof(NRL_RULES_R) / sizeof(ttv_rule) },
    { NRL_RULES_S,      sizeof(NRL_RULES_S) / sizeof(ttv_rule) },
    { NRL_RULES_T,      sizeof(NRL_RULES_T) / sizeof(ttv_rule) },
    { NRL_RULES_U,      sizeof(NRL_RULES_U) / sizeof(ttv_rule) },
    { NRL_RULES_V,      sizeof(NRL_RULES_V) / sizeof(ttv_rule) },
    { NRL_RULES_W,      sizeof(NRL_RULES_W) / sizeof(ttv_rule) },
    { NRL_RULES_X,      sizeof(NRL_RULES_X) / sizeof(ttv_rule) },
    { NRL_RULES_Y,      sizeof(NRL_RULES_Y) / sizeof(ttv_rule) },
    { NRL_RULES_Z,      sizeof(NRL_RULES_Z) / sizeof(ttv_rule) },
};

// ARPABET -> SC-01 phone names, first match wins.  `left`/`right` are the
// adjacent ARPABET symbols; an empty string means "any".  `out` is a
// space-separated list of SC-01 phone names (see TTV_PHONE_NAMES).
//
// A1/A2 note: Geczy's sc01.dll, which these strings were recovered from,
// carries a phone name table with A1 and A2 transposed, so its own lookup resolved the "A1"
// below to phone 0x05.  The ROM disagrees: within every vowel family a higher
// digit is a shorter phone (EH3 19 < EH2 23 < EH1 38 < EH 58, and likewise for
// I and UH), which makes 0x05 = A2 and 0x06 = A1 as the datasheet chart has
// them.  The names here are kept verbatim and resolved through the correct
// table, on the reading that the transposition was the driver's own
// transcription slip and this map is the older, datasheet-named data.  The
// stake is small either way -- a 103 ms vs 71 ms variant of the same vowel, in
// the one context of EY after /l/.

const ttv_arpa_map TTV_ARPABET[] = {
    { "",     "IY",  "",     "E" },
    { "",     "IH",  "",     "I" },
    { "L",    "EY",  "R",    "UH3 A1 I3" },
    { "L",    "EY",  "",     "UH3 A1 AY" },
    { "",     "EY",  "R",    "A I3" },
    { "",     "EY",  "",     "A AY" },
    { "L",    "EH",  "",     "UH3 EH" },
    { "",     "EH",  "",     "EH" },
    { "L",    "AE",  "R",    "UH3 AE EH3" },
    { "L",    "AE",  "",     "UH3 AE" },
    { "",     "AE",  "R",    "AE1 EH3" },
    { "",     "AE",  "",     "AE" },
    { "",     "AA",  "",     "AH" },
    { "L",    "AO",  "R",    "UH3 O" },
    { "L",    "AO",  "ER",   "UH3 AW O2" },
    { "L",    "AO",  "",     "UH3 AW" },
    { "",     "AO",  "R",    "O" },
    { "",     "AO",  "ER",   "AW O2" },
    { "",     "AO",  "",     "AW" },
    { "L",    "OW",  "",     "UH3 O1 U1" },
    { "",     "OW",  "",     "O1 U1" },
    { "L",    "UH",  "",     "UH3 OO" },
    { "",     "UH",  "",     "OO" },
    { "",     "UW",  "",     "IU U" },
    { "IY",   "ER",  "",     "I3 ER" },
    { "ER",   "ER",  "",     "IU R" },
    { "L",    "ER",  "",     "UH3 ER" },
    { "",     "ER",  "L",    "UH3 ER" },
    { "R",    "ER",  "",     "UH3 R" },
    { "",     "ER",  "",     "ER" },
    { "",     "AX",  "",     "UH2" },
    { "",     "AH",  "",     "UH" },
    { "",     "AY",  "L",    "AH AY" },
    { "",     "AY",  "R",    "AH I3" },
    { "",     "AY",  "ER",   "AH AY" },
    { "",     "AY",  "",     "AH E1" },
    { "",     "AW",  "",     "AH O1" },
    { "L",    "OY",  "ER",   "UH3 O1 AY" },
    { "L",    "OY",  "L",    "UH3 O1 AY" },
    { "L",    "OY",  "R",    "UH3 O1 EH2" },
    { "",     "OY",  "ER",   "O1 AY" },
    { "",     "OY",  "L",    "O1 AY" },
    { "",     "OY",  "R",    "O1 EH2" },
    { "",     "OY",  "",     "O1 E1" },
    { "",     "Y",   "",     "Y1" },
    { "",     "P",   "",     "P" },
    { "",     "B",   "",     "B" },
    { "",     "T",   "",     "T" },
    { "",     "D",   "",     "D" },
    { "",     "K",   "",     "K" },
    { "",     "G",   "",     "G" },
    { "",     "F",   "",     "F" },
    { "",     "V",   "",     "V" },
    { "",     "TH",  "",     "TH" },
    { "",     "DH",  "",     "THV" },
    { "",     "S",   "",     "S" },
    { "",     "Z",   "",     "Z" },
    { "",     "SH",  "",     "SH" },
    { "",     "ZH",  "",     "ZH" },
    { "",     "HH",  "",     "H" },
    { "",     "CH",  "",     "T CH" },
    { "",     "JH",  "",     "D J" },
    { "",     "M",   "",     "M" },
    { "",     "N",   "",     "N" },
    { "",     "NX",  "",     "NG" },
    { "IY",   "L",   "",     "I3 L" },
    { "EY",   "L",   "",     "I3 L" },
    { "AY",   "L",   "",     "I3 L" },
    { "OY",   "L",   "",     "I3 L" },
    { "AE",   "L",   "",     "UH3 L" },
    { "AO",   "L",   "",     "UH3 L" },
    { "OW",   "L",   "",     "UH3 L" },
    { "",     "L",   "",     "L" },
    { "",     "W",   "",     "W" },
    { "",     "WH",  "",     "H W" },
    { "",     "R",   "L",    "UH3 R" },
    { "",     "R",   "",     "R" },
    { "",     "_",   "",     "PA0" },
    { "",     ",",   "",     "PA1" },
    { "",     ".",   "",     "PA1 PA1" },
    { "",     "-",   "",     "PA1" },
};

// Words the rules get wrong, rewritten before the rules run -- Tamas Geczy's
// measured exception dictionary (exceptions.c in votraxsc01-nvda,
// BSD-3-Clause).  Each pair is {as written, as respelled}; the text is space-padded so the leading and
// trailing blanks act as word boundaries.
const char *const TTV_EXCEPTIONS[][2] = {
    { " SEARCH ",      " SURCH " },
    { " SEARCHES ",    " SURCHES " },
    { " SEARCHED ",    " SURCHED " },
    { " SEARCHING ",   " SURCHING " },
    { " RESEARCH ",    " RESURCH " },
    { " HEARD ",       " HURD " },
    { " HEARSE ",      " HURSE " },
    { " REHEARSE ",    " REHURSE " },
    { " REHEARSAL ",   " REHURSAL " },
    { " BEAR ",        " BAIR " },
    { " BEARS ",       " BAIRS " },
    { " WEAR ",        " WAIR " },
    { " WEARS ",       " WAIRS " },
    { " SWEAR ",       " SWAIR " },
    { " SWEARS ",      " SWAIRS " },
    { " PEAR ",        " PAIR " },
    { " PEARS ",       " PAIRS " },
};

// Number names, as ARPABET.  [0..19] are zero..nineteen and [20..27] are
// twenty, thirty .. ninety; TTV_ORDINALS has the same shape.
const char *const TTV_CARDINALS[28] = {
    "zIHrOW", "wAHn", "tUW", "THrIY",
    "fOWr", "fAYv", "sIHks", "sEHvAXn",
    "EYt", "nAYn", "tEHn", "IYlEHvAXn",
    "twEHlv", "THERtIYn", "fOWrtIYn", "fIHftIYn",
    "sIHkstIYn", "sEHvEHntIYn", "EYtIYn", "nAYntIYn",
    "twEHntIY", "THERtIY", "fAOrtIY", "fIHftIY",
    "sIHkstIY", "sEHvEHntIY", "EYtIY", "nAYntIY",
};

const char *const TTV_ORDINALS[28] = {
    "zIHrOWEHTH", "fERst", "sEHkAHnd", "THERd",
    "fOWrTH", "fIHfTH", "sIHksTH", "sEHvEHnTH",
    "EYtTH", "nAYnTH", "tEHnTH", "IYlEHvEHnTH",
    "twEHlvTH", "THERtIYnTH", "fAOrtIYnTH", "fIHftIYnTH",
    "sIHkstIYnTH", "sEHvEHntIYnTH", "EYtIYnTH", "nAYntIYnTH",
    "twEHntIYEHTH", "THERtIYEHTH", "fOWrtIYEHTH", "fIHftIYEHTH",
    "sIHkstIYEHTH", "sEHvEHntIYEHTH", "EYtIYEHTH", "nAYntIYEHTH",
};

// Spoken names for the 128 ASCII codes, as ARPABET -- what a screen reader
// says for a lone character.  Index by the character's own code.
const char *const TTV_ASCII_NAMES[128] = {
    /*   0 */ "nUWl",                        /*   1 */ "stAArt AXv hEHdER",
    /*   2 */ "stAArt AXv tEHkst",           /*   3 */ "EHnd AXv tEHkst",
    /*   4 */ "EHnd AXv trAEnsmIHSHAXn",     /*   5 */ "EHnkwAYr",
    /*   6 */ "AEk",                         /*   7 */ "bEHl",
    /*   8 */ "bAEkspEYs",                   /*   9 */ "tAEb",
    /*  10 */ "lIHnIYfIYd",                  /*  11 */ "vERtIHkAXl tAEb",
    /*  12 */ "fAOrmfIYd",                   /*  13 */ "kAErAYj rIYtERn",
    /*  14 */ "SHIHft AWt",                  /*  15 */ "SHIHft IHn",
    /*  16 */ "dIHlIYt",                     /*  17 */ "dIHvIHs kAAntrAAl wAHn",
    /*  18 */ "dIHvIHs kAAntrAAl tUW",       /*  19 */ "dIHvIHs kAAntrAAl THrIY",
    /*  20 */ "dIHvIHs kAAntrAAl fOWr",      /*  21 */ "nAEk",
    /*  22 */ "sIHnk",                       /*  23 */ "EHnd tEHkst blAAk",
    /*  24 */ "kAEnsEHl",                    /*  25 */ "EHnd AXv mEHsIHj",
    /*  26 */ "sUWbstIHtUWt",                /*  27 */ "EHskEYp",
    /*  28 */ "fAYEHld sIYpERAEtER",         /*  29 */ "grUWp sIYpERAEtER",
    /*  30 */ "rIYkAOrd sIYpERAEtER",        /*  31 */ "yUWnIHt sIYpERAEtER",
    /*  32 */ "spEYs",                       /*  33 */ "EHksklAEmEYSHAXn mAArk",
    /*  34 */ "dAHbl kwOWt",                 /*  35 */ "nUWmbER sAYn",
    /*  36 */ "dAAlER sAYn",                 /*  37 */ "pERsEHnt",
    /*  38 */ "AEmpERsAEnd",                 /*  39 */ "kwOWt",
    /*  40 */ "OWpEHn pEHrEHn",              /*  41 */ "klOWz pEHrEHn",
    /*  42 */ "AEstEHrIHsk",                 /*  43 */ "plAHs",
    /*  44 */ "kAAmmAX",                     /*  45 */ "mIHnAHs",
    /*  46 */ "pIYrIYAAd",                   /*  47 */ "slAESH",
    /*  48 */ "zIHrOW",                      /*  49 */ "wAHn",
    /*  50 */ "tUW",                         /*  51 */ "THrIY",
    /*  52 */ "fOWr",                        /*  53 */ "fAYv",
    /*  54 */ "sIHks",                       /*  55 */ "sEHvAXn",
    /*  56 */ "EYt",                         /*  57 */ "nAYn",
    /*  58 */ "kAAlAXn",                     /*  59 */ "sEHmIHkAAlAXn",
    /*  60 */ "lEHs DHAEn",                  /*  61 */ "EHkwAXl sAYn",
    /*  62 */ "grEYtER DHAEn",               /*  63 */ "kwEHsCHAXn mAArk",
    /*  64 */ "AEt sAYn",                    /*  65 */ "EY",
    /*  66 */ "bIY",                         /*  67 */ "sIY",
    /*  68 */ "dIY",                         /*  69 */ "IY",
    /*  70 */ "EHf",                         /*  71 */ "jIY",
    /*  72 */ "EYCH",                        /*  73 */ "AY",
    /*  74 */ "jEY",                         /*  75 */ "kEY",
    /*  76 */ "EHl",                         /*  77 */ "EHm",
    /*  78 */ "EHn",                         /*  79 */ "OW",
    /*  80 */ "pIY",                         /*  81 */ "kyUW",
    /*  82 */ "AAr",                         /*  83 */ "EHs",
    /*  84 */ "tIY",                         /*  85 */ "yUW",
    /*  86 */ "vIY",                         /*  87 */ "dAHblyUW",
    /*  88 */ "EHks",                        /*  89 */ "wAY",
    /*  90 */ "zIY",                         /*  91 */ "lEHft brAEkEHt",
    /*  92 */ "bAEkslAESH",                  /*  93 */ "rAYt brAEkEHt",
    /*  94 */ "kAErEHt",                     /*  95 */ "AHndERskAOr",
    /*  96 */ "AEpAAstrAAfIH",               /*  97 */ "EY",
    /*  98 */ "bIY",                         /*  99 */ "sIY",
    /* 100 */ "dIY",                         /* 101 */ "IY",
    /* 102 */ "EHf",                         /* 103 */ "jIY",
    /* 104 */ "EYCH",                        /* 105 */ "AY",
    /* 106 */ "jEY",                         /* 107 */ "kEY",
    /* 108 */ "EHl",                         /* 109 */ "EHm",
    /* 110 */ "EHn",                         /* 111 */ "OW",
    /* 112 */ "pIY",                         /* 113 */ "kyUW",
    /* 114 */ "AAr",                         /* 115 */ "EHs",
    /* 116 */ "tIY",                         /* 117 */ "yUW",
    /* 118 */ "vIY",                         /* 119 */ "dAHblyUW",
    /* 120 */ "EHks",                        /* 121 */ "wAY",
    /* 122 */ "zIY",                         /* 123 */ "lEHft brEYs",
    /* 124 */ "vERtIHkAXl bAAr",             /* 125 */ "rAYt brEYs",
    /* 126 */ "tAYld",                       /* 127 */ "dEHl",
};

// Abbreviations expanded before anything else runs, as whole words on
// space-padded text.  Short, and the same three every Votrax-era front end
// carried.  " PHD " was also recognised but has no expansion -- it was spelled
// out letter by letter, which is still the right answer.
const char *const TTV_ABBREVIATIONS[][2] = {
    { " DR ",   " DOCTOR " },
    { " MR ",   " MISTER " },
    { " MRS ",  " MISSUS " },
};

// Scale words, spelled as Wasser's 1985 saynum.c spells them (public domain).
// An ordinal appends TH to whichever of these ends the number.
const char *const TTV_HUNDRED = "hAHndrEHd";
const char *const TTV_THOUSAND = "THAWzAEnd";
const char *const TTV_MILLION = "mIHlIYAXn";
const char *const TTV_BILLION = "bIHlIYAXn";

// Words for reading amounts and decimals, as ARPABET: "3.14" is three POINT
// one four, "$4.20" is four DOLLARS AND twenty CENTS.
const char *const TTV_POINT = "pOYnt";
const char *const TTV_DOLLAR = "dAAlER";
const char *const TTV_DOLLARS = "dAAlERz";
const char *const TTV_AND = "AAnd";
const char *const TTV_CENT = "sEHnt";
const char *const TTV_CENTS = "sEHnts";

// The 64 SC-01 phone names, indexed by phone code.  Datasheet chart order;
// note 0x05 = A2 and 0x06 = A1.  Geczy's sc01.dll has those two swapped, which the
// ROM's own duration fields disprove -- see docs/tech-overview.md, Part 4.
const char *const TTV_PHONE_NAMES[64] = {
    "EH3",   "EH2",   "EH1",   "PA0",   "DT",    "A2",    "A1",    "ZH",   
    "AH2",   "I3",    "I2",    "I1",    "M",     "N",     "B",     "V",    
    "CH",    "SH",    "Z",     "AW1",   "NG",    "AH1",   "OO1",   "OO",   
    "L",     "K",     "J",     "H",     "G",     "F",     "D",     "S",    
    "A",     "AY",    "Y1",    "UH3",   "AH",    "P",     "O",     "I",    
    "U",     "Y",     "T",     "R",     "E",     "W",     "AE",    "AE1",  
    "AW2",   "UH2",   "UH1",   "UH",    "O2",    "O1",    "IU",    "U1",   
    "THV",   "TH",    "ER",    "EH",    "E1",    "AW",    "PA1",   "STOP", 
};

/* Counts for the tables whose length the matcher needs and C will not tell
 * it.  Defined here rather than in the header so that adding a row to a table
 * cannot leave a stale count behind in a different file. */
const size_t TTV_ARPABET_COUNT = sizeof TTV_ARPABET / sizeof TTV_ARPABET[0];
const size_t TTV_EXCEPTION_COUNT = sizeof TTV_EXCEPTIONS / sizeof TTV_EXCEPTIONS[0];
const size_t TTV_ABBREVIATION_COUNT =
    sizeof TTV_ABBREVIATIONS / sizeof TTV_ABBREVIATIONS[0];

const char *ttv_phone_name(int code)
{
    return (code >= 0 && code < 64) ? TTV_PHONE_NAMES[code] : 0;
}
