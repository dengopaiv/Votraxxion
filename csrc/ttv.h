// Text to SC-01 phone codes — the front end, in C++ with no dependencies.
//
// Two stages, walking the tables in ttv_tables.h:
//
//   text  --NRL_RULES-->  ARPABET  --ARPABET_TO_SC01-->  phone codes
//
// Nothing here allocates beyond the output strings, nothing reads a file, and
// nothing needs a dictionary: nine hundred-odd table entries and about two
// hundred lines of matcher are the whole English front end.  That is the point
// — an NVDA add-on gets a native library it can drive over a few C calls, with
// no Python and no data files to install alongside it.
//
// See docs/tech-overview.md, Part 4, for the rule format and the table
// provenance.
#pragma once

#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "ttv_tables.h"

namespace ttv {

// ---------------------------------------------------------------- classes --
//
// The character classes the NRL context syntax is written in.  `isvoiced` is
// the set the rules spell `.`; it is not the phonetic voiced set, it is the
// specific eleven letters the 1976 report chose.

inline bool is_vowel(char c) {
    return c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U';
}

inline bool is_consonant(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) && !is_vowel(c);
}

inline bool is_voiced(char c) {
    return std::strchr("BDVGJLMNRWZ", c) != nullptr && c != '\0';
}

inline bool is_front_vowel(char c) {
    return c == 'E' || c == 'I' || c == 'Y';
}

// The rules spell a word boundary as a space, but they are applied to running
// text where the boundary is just as likely to be a comma or a full stop.
// Treating any non-letter as a boundary is what makes "hello," come out the
// same as "hello" -- matching on a literal space instead loses the rules that
// need to see the end of the word, and the vowel falls through to a wrong
// default.
inline bool is_boundary(char c) {
    return !std::isalnum(static_cast<unsigned char>(c));
}

// ---------------------------------------------------------- context match --
//
// Both contexts are matched outward from the cursor.  The left pattern is
// stored in reverse reading order, so it is walked from its own end backwards
// while the text walks backwards too; the right pattern is walked forwards.
// `text` is the whole space-padded buffer and `pos` the index just outside the
// matched letters, so both walks stay in bounds on the padding.

inline bool match_left(const char *pattern, const std::string &text, int pos) {
    if (*pattern == '\0') return true;
    int pat = static_cast<int>(std::strlen(pattern)) - 1;
    for (; pat >= 0; pat--) {
        char p = pattern[pat];
        char t = pos >= 0 ? text[pos] : ' ';
        if (std::isalpha(static_cast<unsigned char>(p))) {
            if (p != t) return false;
            pos--;
            continue;
        }
        switch (p) {
        case ' ':
            if (!is_boundary(t)) return false;
            pos--;
            break;
        case '#':  // one or more vowels
            if (!is_vowel(t)) return false;
            pos--;
            while (pos >= 0 && is_vowel(text[pos])) pos--;
            break;
        case ':':  // zero or more consonants
            while (pos >= 0 && is_consonant(text[pos])) pos--;
            break;
        case '^':  // exactly one consonant
            if (!is_consonant(t)) return false;
            pos--;
            break;
        case '.':  // a voiced consonant
            if (!is_voiced(t)) return false;
            pos--;
            break;
        case '+':  // a front vowel
            if (!is_front_vowel(t)) return false;
            pos--;
            break;
        default:
            return false;  // '%' is right-context only; anything else is a typo
        }
    }
    return true;
}

inline bool match_right(const char *pattern, const std::string &text, int pos) {
    if (*pattern == '\0') return true;
    const int n = static_cast<int>(text.size());
    for (const char *pat = pattern; *pat; pat++) {
        char t = pos < n ? text[pos] : ' ';
        if (std::isalpha(static_cast<unsigned char>(*pat))) {
            if (*pat != t) return false;
            pos++;
            continue;
        }
        switch (*pat) {
        case ' ':
            if (!is_boundary(t)) return false;
            pos++;
            break;
        case '#':
            if (!is_vowel(t)) return false;
            pos++;
            while (pos < n && is_vowel(text[pos])) pos++;
            break;
        case ':':
            while (pos < n && is_consonant(text[pos])) pos++;
            break;
        case '^':
            if (!is_consonant(t)) return false;
            pos++;
            break;
        case '.':
            if (!is_voiced(t)) return false;
            pos++;
            break;
        case '+':
            if (!is_front_vowel(t)) return false;
            pos++;
            break;
        case '%': {
            // A suffix: E, ER, ES, ED, ELY or ING.  The one context class that
            // consumes a variable, spelled-out string rather than a class.
            auto at = [&](int i) { return i < n ? text[i] : ' '; };
            if (t == 'E') {
                pos++;
                if (at(pos) == 'L') {
                    if (at(pos + 1) != 'Y') return false;
                    pos += 2;
                } else if (at(pos) == 'R' || at(pos) == 'S' || at(pos) == 'D') {
                    pos++;
                }
                break;
            }
            if (t == 'I' && at(pos + 1) == 'N' && at(pos + 2) == 'G') {
                pos += 3;
                break;
            }
            return false;
        }
        default:
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------- stage one ---

// Spell out a non-negative integer below 1000 using CARDINALS.  Larger runs of
// digits are read digit by digit, which is what a screen reader wants for
// things that are not really numbers (version strings, IDs, phone numbers).
inline void append_number(std::string &out, const std::string &digits) {
    auto say = [&](const char *word) {
        out += word;
        out += ' ';
    };
    if (digits.size() > 3) {
        for (char c : digits) say(CARDINALS[c - '0']);
        return;
    }
    int value = std::stoi(digits);
    if (value >= 100) {
        say(CARDINALS[value / 100]);
        say("hAHndrEHd");
        value %= 100;
        if (value == 0) return;
    }
    if (value < 20) {
        say(CARDINALS[value]);
    } else {
        say(CARDINALS[18 + value / 10]);   // 20 -> index 20, 30 -> 21, ...
        if (value % 10) say(CARDINALS[value % 10]);
    }
}

// Whole-word rewrites applied before the rules run: abbreviations expanded,
// then the respellings for words the rules get wrong.  The rules cannot see
// far enough ahead for either, so the front end rewrites the word into
// something they do handle.  Order matters -- " DR " becomes " DOCTOR " first,
// so the expansion is then subject to the ordinary rules.
inline void replace_all(std::string &text, const std::string &from,
                        const std::string &to) {
    if (from.empty()) return;
    for (std::string::size_type at = text.find(from);
         at != std::string::npos;
         at = text.find(from, at + to.size())) {
        text.replace(at, from.size(), to);
    }
}

inline std::string apply_exceptions(std::string text) {
    for (const auto &pair : ABBREVIATIONS) replace_all(text, pair[0], pair[1]);
    for (const auto &pair : NRL_EXCEPTIONS) replace_all(text, pair[0], pair[1]);
    return text;
}

// English spelling to ARPABET.  Upper-case letters spell out two-letter
// ARPABET symbols; lower-case letters are single-letter consonants.  That
// convention is the tables', not ours, and the tokenizer below relies on it.
inline std::string to_arpabet(const std::string &text) {
    std::string in = " ";
    for (unsigned char c : text) in += static_cast<char>(std::toupper(c));
    in += ' ';
    in = apply_exceptions(in);

    std::string out;
    int pos = 1;
    const int n = static_cast<int>(in.size());
    while (pos < n - 1) {
        char c = in[pos];

        if (std::isdigit(static_cast<unsigned char>(c))) {
            int end = pos;
            while (end < n && std::isdigit(static_cast<unsigned char>(in[end]))) end++;
            append_number(out, in.substr(pos, end - pos));
            pos = end;
            continue;
        }

        const TtvRuleGroup &group =
            std::isalpha(static_cast<unsigned char>(c))
                ? NRL_RULES[1 + (c - 'A')]
                : NRL_RULES[0];

        bool matched = false;
        for (std::size_t i = 0; i < group.count; i++) {
            const TtvRule &rule = group.rules[i];
            const int len = static_cast<int>(std::strlen(rule.match));
            if (in.compare(pos, len, rule.match) != 0) continue;
            if (!match_left(rule.left, in, pos - 1)) continue;
            if (!match_right(rule.right, in, pos + len)) continue;
            out += rule.out;
            pos += len;
            matched = true;
            break;
        }
        if (!matched) pos++;   // no rule: drop the character rather than stall
    }
    return out;
}

// ------------------------------------------------------------- stage two ---

// Split an ARPABET string into symbols.  Two upper-case letters that name a
// symbol in the map are taken together; anything else is a single character,
// upper-cased, so the rules' lower-case consonants ("grEYt") land on the map's
// upper-case keys.
inline bool is_map_key(const std::string &sym) {
    for (const ArpaMap &m : ARPABET_TO_SC01)
        if (sym == m.arpa) return true;
    return false;
}

inline bool is_two_letter_symbol(const char *s) {
    if (is_map_key(std::string(s, 2))) return true;
    // NG is what the letter-to-sound rules emit for the velar nasal; the map
    // spells the same sound NX.  Without this alias "young" comes out as
    // /n/ + /g/, so treat the two as one symbol and translate below.
    return s[0] == 'N' && s[1] == 'G';
}

// The rules write /h/ and /dZ/ as the single letters h and j, but the map
// keys them as HH and JH.  Nothing else in either table is spelled two ways,
// so the rule is simply: a lone consonant that is not a key, but becomes one
// with an H after it, is that symbol.  Without this, /h/ and /dZ/ are silently
// dropped and "hello" loses its H.
inline std::string canonical_symbol(const std::string &sym) {
    if (sym == "NG") return "NX";
    if (is_map_key(sym)) return sym;
    if (sym.size() == 1 && is_map_key(sym + "H")) return sym + "H";
    return sym;
}

inline std::vector<std::string> tokenize_arpabet(const std::string &arpa) {
    std::vector<std::string> out;
    for (std::string::size_type i = 0; i < arpa.size();) {
        if (i + 1 < arpa.size() &&
            std::isupper(static_cast<unsigned char>(arpa[i])) &&
            std::isupper(static_cast<unsigned char>(arpa[i + 1])) &&
            is_two_letter_symbol(arpa.c_str() + i)) {
            out.push_back(canonical_symbol(arpa.substr(i, 2)));
            i += 2;
            continue;
        }
        char c = static_cast<char>(std::toupper(static_cast<unsigned char>(arpa[i])));
        out.push_back(canonical_symbol(std::string(1, c == ' ' ? '_' : c)));
        i++;
    }
    return out;
}

inline int phone_by_name(const std::string &name) {
    for (int i = 0; i < 64; i++)
        if (name == PHONE_NAMES[i]) return i;
    return -1;
}

// ARPABET to SC-01 phone codes.  For each symbol the first map entry whose
// left and right contexts hold wins; the entries are ordered so that every
// symbol ends with an unconditional fallback, and lookup cannot fail.
inline std::vector<std::uint8_t> arpabet_to_phones(const std::string &arpa) {
    const std::vector<std::string> syms = tokenize_arpabet(arpa);
    std::vector<std::uint8_t> phones;
    for (std::size_t i = 0; i < syms.size(); i++) {
        const std::string &left = i ? syms[i - 1] : std::string();
        const std::string &right = i + 1 < syms.size() ? syms[i + 1] : std::string();
        for (const ArpaMap &m : ARPABET_TO_SC01) {
            if (syms[i] != m.arpa) continue;
            if (*m.left && left != m.left) continue;
            if (*m.right && right != m.right) continue;
            // out is a space-separated list of SC-01 phone names
            const char *p = m.out;
            while (*p) {
                const char *end = std::strchr(p, ' ');
                std::string name = end ? std::string(p, end) : std::string(p);
                int code = phone_by_name(name);
                if (code >= 0) phones.push_back(static_cast<std::uint8_t>(code));
                if (!end) break;
                p = end + 1;
            }
            break;
        }
    }
    return phones;
}

// -------------------------------------------------------------- prosody ---
//
// The SC-01's pitch input is two bits: four levels, nothing between.  Measured
// on our own core at the datasheet clock they are 78, 89, 104 and 125 Hz — a
// range of about a fifth, in four steps, and that is the entire pitch budget.
//
// It is not much, but flat speech is the single most fatiguing thing about
// early synthesizers, and four levels are enough to carry the one contour that
// matters most: declination.  English statements drift downward in pitch across
// a clause and drop at the end; questions do the opposite and rise.  A listener
// uses that fall to hear where a sentence ends, which is why flat output makes
// running text feel like it never stops.
//
// So: assign a level per phone from its position within its own sentence, and
// let the sentence's final punctuation choose the shape.  No stress model, no
// syllable model — the letter-to-sound rules do not give us either — but the
// clause-level contour is the part that four levels can actually express.

//: A phone code and its inflection travel together in one byte: the phone in
//: bits 0-5, the level in bits 6-7.  A byte whose top bits are clear is
//: therefore just a plain phone code, which is what makes the packed and
//: unpacked forms interchangeable everywhere a phone is accepted.
constexpr std::uint8_t PHONE_MASK = 0x3F;
constexpr int INFLECTION_SHIFT = 6;

//: The level a contour treats as neutral.  Levels are stored absolutely, but
//: a caller's own pitch setting shifts the whole contour relative to this, so
//: that a caller who sets "normal" gets exactly the contour computed here.
constexpr int NEUTRAL_INFLECTION = 1;

inline std::uint8_t pack(int phone, int inflection) {
    return static_cast<std::uint8_t>((phone & PHONE_MASK) |
                                     ((inflection & 0x03) << INFLECTION_SHIFT));
}

inline int phone_of(std::uint8_t packed) { return packed & PHONE_MASK; }

inline int inflection_of(std::uint8_t packed) {
    return (packed >> INFLECTION_SHIFT) & 0x03;
}

inline bool is_pause_phone(int phone) {
    return phone == 0x03 || phone == 0x3E;   // PA0, PA1
}

enum class Sentence { Statement, Question, Exclamation };

inline Sentence classify(char terminator) {
    if (terminator == '?') return Sentence::Question;
    if (terminator == '!') return Sentence::Exclamation;
    return Sentence::Statement;
}

// The level for a phone `position` of the way through its sentence.
//
// Statements start a step above neutral and fall to the floor — the final drop
// is what makes a sentence sound finished.  Questions do the reverse, sitting
// at neutral and climbing over the last third.  Exclamations start high and
// fall, but stop short of the floor so they keep some energy.
inline int contour_level(Sentence type, double position) {
    switch (type) {
    case Sentence::Question:
        if (position < 0.55) return 1;
        if (position < 0.80) return 2;
        return 3;
    case Sentence::Exclamation:
        if (position < 0.20) return 3;
        if (position < 0.55) return 2;
        return 1;
    case Sentence::Statement:
    default:
        if (position < 0.25) return 2;
        if (position < 0.70) return 1;
        return 0;
    }
}

// Stamp a contour across one sentence's phones, in place.
//
// Pauses are skipped when measuring position — they are silent, so counting
// them would let a comma-heavy sentence spend its whole contour on gaps — but
// they still carry the level of whatever preceded them, so the packed stream
// never has a level discontinuity in it.
inline void apply_contour(std::vector<std::uint8_t> &phones, Sentence type) {
    std::size_t voiced = 0;
    for (std::uint8_t b : phones)
        if (!is_pause_phone(phone_of(b))) voiced++;
    if (voiced == 0) return;

    std::size_t seen = 0;
    int level = contour_level(type, 0.0);
    for (std::uint8_t &b : phones) {
        const int phone = phone_of(b);
        if (!is_pause_phone(phone)) {
            const double position =
                voiced > 1 ? static_cast<double>(seen) / (voiced - 1) : 0.0;
            level = contour_level(type, position);
            seen++;
        }
        b = pack(phone, level);
    }
}

// ------------------------------------------------------------------- API ---

//: The phone the chip idles on, and what closes an utterance.
constexpr std::uint8_t STOP = 0x3F;

//: A word gap.
constexpr std::uint8_t PA0 = 0x03;

// Speak a run of English text, flat — every phone at inflection 0.  Use this
// when the caller means to supply its own contour.
inline std::vector<std::uint8_t> translate_flat(const std::string &text) {
    return arpabet_to_phones(to_arpabet(text));
}

// Split text into sentences at . ? or ! followed by whitespace or the end.
//
// The trailing-whitespace requirement is what keeps "3.14" and "1.5" in one
// piece. It does not save "e.g." or "Dr." — an abbreviation ending in a full
// stop still reads as a sentence end, which costs a contour reset and nothing
// worse.
inline std::vector<std::string> split_sentences(const std::string &text) {
    std::vector<std::string> out;
    std::string::size_type start = 0;
    for (std::string::size_type i = 0; i < text.size(); i++) {
        const char c = text[i];
        if (c != '.' && c != '?' && c != '!') continue;
        const bool at_end = i + 1 >= text.size();
        if (!at_end && !std::isspace(static_cast<unsigned char>(text[i + 1]))) continue;
        out.push_back(text.substr(start, i - start + 1));
        start = i + 1;
    }
    if (start < text.size()) out.push_back(text.substr(start));
    return out;
}

// Speak a run of English text, with a pitch contour over each sentence.
//
// Returns packed bytes: use phone_of() and inflection_of(), or hand them
// straight to the scheduler, which understands them.
inline std::vector<std::uint8_t> translate(const std::string &text) {
    std::vector<std::uint8_t> out;
    for (const std::string &sentence : split_sentences(text)) {
        std::vector<std::uint8_t> phones = translate_flat(sentence);
        if (phones.empty()) continue;
        char terminator = ' ';
        for (std::string::size_type i = sentence.size(); i-- > 0;) {
            if (!std::isspace(static_cast<unsigned char>(sentence[i]))) {
                terminator = sentence[i];
                break;
            }
        }
        apply_contour(phones, classify(terminator));
        out.insert(out.end(), phones.begin(), phones.end());
    }
    return out;
}

// Spell text out character by character, the way a screen reader reads a
// password field or an unfamiliar word.  Control codes get their names too.
// Deliberately flat: a spelled-out string is a list, not a sentence, and a
// declination contour over it would imply a shape it does not have.
inline std::vector<std::uint8_t> spell(const std::string &text) {
    std::string arpa;
    for (unsigned char c : text) {
        if (c > 127) continue;
        arpa += ASCII_NAMES[c];
        arpa += ' ';
    }
    std::vector<std::uint8_t> phones = arpabet_to_phones(arpa);
    for (std::uint8_t &b : phones) b = pack(phone_of(b), NEUTRAL_INFLECTION);
    return phones;
}

}  // namespace ttv
