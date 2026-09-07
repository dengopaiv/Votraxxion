/* The English front end -- see ttv.h. */

#include "ttv.h"

#include <string.h>

#include "ttv_tables.h"

/* ---------------------------------------------------------------- classes --
 *
 * The character classes the NRL context syntax is written in.  `is_voiced` is
 * the set the rules spell `.`; it is not the phonetic voiced set, it is the
 * specific eleven letters the 1976 report chose.
 *
 * These test ASCII ranges directly where the C++ called isalpha/isalnum on an
 * unsigned char.  In the C locale the two agree exactly, and the working text
 * has already been upper-cased, so this is the same classification -- but it
 * also removes a latent hazard: under a locale where isalpha() is true for a
 * byte above 127, `NRL_RULES[1 + (c - 'A')]` indexes past the end of a 27-entry
 * array. */

static int is_upper(char c) { return c >= 'A' && c <= 'Z'; }
static int is_digit(char c) { return c >= '0' && c <= '9'; }
static int is_alnum(char c) { return is_upper(c) || is_digit(c) ||
                                     (c >= 'a' && c <= 'z'); }

static int is_vowel(char c)
{
    return c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U';
}

static int is_consonant(char c)
{
    return is_upper(c) && !is_vowel(c);
}

static int is_voiced(char c)
{
    return c != '\0' && strchr("BDVGJLMNRWZ", c) != NULL;
}

static int is_front_vowel(char c)
{
    return c == 'E' || c == 'I' || c == 'Y';
}

/* The rules spell a word boundary as a space, but they are applied to running
 * text where the boundary is just as likely to be a comma or a full stop.
 * Treating any non-alphanumeric as a boundary is what makes "hello," come out
 * the same as "hello" -- matching on a literal space instead loses the rules
 * that need to see the end of the word, and the vowel falls through to a wrong
 * default. */
static int is_boundary(char c)
{
    return !is_alnum(c);
}

static char to_upper(char c)
{
    return (c >= 'a' && c <= 'z') ? (char)(c - 'a' + 'A') : c;
}

/* ---------------------------------------------------------- context match --
 *
 * Both contexts are matched outward from the cursor.  The left pattern is
 * stored in reverse reading order, so it is walked from its own end backwards
 * while the text walks backwards too; the right pattern is walked forwards.
 * `text` is the whole space-padded buffer and `pos` the index just outside the
 * matched letters, so both walks stay in bounds on the padding. */

static int match_left(const char *pattern, const char *text, int pos)
{
    int pat;
    if (*pattern == '\0')
        return 1;
    for (pat = (int)strlen(pattern) - 1; pat >= 0; pat--) {
        char p = pattern[pat];
        char t = pos >= 0 ? text[pos] : ' ';
        if (is_upper(p)) {
            if (p != t)
                return 0;
            pos--;
            continue;
        }
        switch (p) {
        case ' ':
            if (!is_boundary(t)) return 0;
            pos--;
            break;
        case '#':  /* one or more vowels */
            if (!is_vowel(t)) return 0;
            pos--;
            while (pos >= 0 && is_vowel(text[pos])) pos--;
            break;
        case ':':  /* zero or more consonants */
            while (pos >= 0 && is_consonant(text[pos])) pos--;
            break;
        case '^':  /* exactly one consonant */
            if (!is_consonant(t)) return 0;
            pos--;
            break;
        case '.':  /* a voiced consonant */
            if (!is_voiced(t)) return 0;
            pos--;
            break;
        case '+':  /* a front vowel */
            if (!is_front_vowel(t)) return 0;
            pos--;
            break;
        default:
            return 0;  /* '%' is right-context only; anything else is a typo */
        }
    }
    return 1;
}

static int match_right(const char *pattern, const char *text, int n, int pos)
{
    const char *pat;
    if (*pattern == '\0')
        return 1;
    for (pat = pattern; *pat; pat++) {
        char t = pos < n ? text[pos] : ' ';
        if (is_upper(*pat)) {
            if (*pat != t)
                return 0;
            pos++;
            continue;
        }
        switch (*pat) {
        case ' ':
            if (!is_boundary(t)) return 0;
            pos++;
            break;
        case '#':
            if (!is_vowel(t)) return 0;
            pos++;
            while (pos < n && is_vowel(text[pos])) pos++;
            break;
        case ':':
            while (pos < n && is_consonant(text[pos])) pos++;
            break;
        case '^':
            if (!is_consonant(t)) return 0;
            pos++;
            break;
        case '.':
            if (!is_voiced(t)) return 0;
            pos++;
            break;
        case '+':
            if (!is_front_vowel(t)) return 0;
            pos++;
            break;
        case '%': {
            /* A suffix: E, ER, ES, ED, ELY or ING.  The one context class that
             * consumes a variable, spelled-out string rather than a class. */
            char a1 = pos + 1 < n ? text[pos + 1] : ' ';
            char a2 = pos + 2 < n ? text[pos + 2] : ' ';
            if (t == 'E') {
                pos++;
                a1 = pos < n ? text[pos] : ' ';
                if (a1 == 'L') {
                    char nx = pos + 1 < n ? text[pos + 1] : ' ';
                    if (nx != 'Y') return 0;
                    pos += 2;
                } else if (a1 == 'R' || a1 == 'S' || a1 == 'D') {
                    pos++;
                }
                break;
            }
            if (t == 'I' && a1 == 'N' && a2 == 'G') {
                pos += 3;
                break;
            }
            return 0;
        }
        default:
            return 0;
        }
    }
    return 1;
}

/* ---------------------------------------------------------------- buffers --
 *
 * One context on the caller's stack holds every working buffer, so no stage
 * allocates and none of them are static -- two threads can translate at once.
 * About 32 KB, which is nothing on a thread stack and is why these are not
 * heap. */

typedef struct {
    char text[TTV_TEXT_MAX];
    int  text_len;

    char arpa[TTV_ARPA_MAX];
    int  arpa_len;
    int  arpa_full;          /* the length it would have been, unbounded */

    char syms[TTV_SYMS_MAX][4];
    int  sym_count;

    uint8_t phones[TTV_PHONES_MAX];
    size_t  phone_count;     /* full count, may exceed TTV_PHONES_MAX */
} ttv_ctx;

static void arpa_putc(ttv_ctx *x, char c)
{
    x->arpa_full++;
    if (x->arpa_len < TTV_ARPA_MAX - 1)
        x->arpa[x->arpa_len++] = c;
}

static void arpa_puts(ttv_ctx *x, const char *s)
{
    for (; *s; s++)
        arpa_putc(x, *s);
}

/* A word and a trailing space, which is how every table entry is spoken. */
static void arpa_say(ttv_ctx *x, const char *word)
{
    arpa_puts(x, word);
    arpa_putc(x, ' ');
}

static void phone_put(ttv_ctx *x, uint8_t phone)
{
    if (x->phone_count < TTV_PHONES_MAX)
        x->phones[x->phone_count] = phone;
    x->phone_count++;
}

/* ------------------------------------------------------------- stage one --- */

/* Spell out a non-negative integer below 1000 using TTV_CARDINALS.  Longer
 * runs of digits are read digit by digit, which is what a screen reader wants
 * for things that are not really numbers -- version strings, IDs, phone
 * numbers. */
static void append_number(ttv_ctx *x, const char *digits, int len)
{
    int value, i;

    if (len > 3) {
        for (i = 0; i < len; i++)
            arpa_say(x, TTV_CARDINALS[digits[i] - '0']);
        return;
    }

    value = 0;
    for (i = 0; i < len; i++)
        value = value * 10 + (digits[i] - '0');

    if (value >= 100) {
        arpa_say(x, TTV_CARDINALS[value / 100]);
        arpa_say(x, "hAHndrEHd");
        value %= 100;
        if (value == 0)
            return;
    }
    if (value < 20) {
        arpa_say(x, TTV_CARDINALS[value]);
    } else {
        arpa_say(x, TTV_CARDINALS[18 + value / 10]);   /* 20 -> 20, 30 -> 21 */
        if (value % 10)
            arpa_say(x, TTV_CARDINALS[value % 10]);
    }
}

/* Replace every occurrence of `from` with `to`, in place.
 *
 * The scan resumes just past the text that was written, not just past the
 * match -- the same rule std::string::find(from, at + to.size()) followed, so
 * a replacement containing the pattern is not rescanned. */
static void replace_all(ttv_ctx *x, const char *from, const char *to)
{
    int from_len = (int)strlen(from);
    int to_len = (int)strlen(to);
    int at = 0;

    if (from_len == 0)
        return;

    while (at <= x->text_len - from_len) {
        char *hit = strstr(x->text + at, from);
        int pos, tail;
        if (!hit)
            return;
        pos = (int)(hit - x->text);

        tail = x->text_len - pos - from_len;
        if (pos + to_len + tail >= TTV_TEXT_MAX)
            return;                     /* would not fit; leave it alone */
        memmove(x->text + pos + to_len, x->text + pos + from_len,
                (size_t)tail + 1);      /* +1 carries the terminator */
        memcpy(x->text + pos, to, (size_t)to_len);
        x->text_len += to_len - from_len;

        at = pos + to_len;
    }
}

/* Whole-word rewrites applied before the rules run: abbreviations expanded,
 * then the respellings for words the rules get wrong.  The rules cannot see
 * far enough ahead for either, so the front end rewrites the word into
 * something they do handle.  Order matters -- " DR " becomes " DOCTOR " first,
 * so the expansion is then subject to the ordinary rules. */
static void apply_exceptions(ttv_ctx *x)
{
    size_t i;
    for (i = 0; i < TTV_ABBREVIATION_COUNT; i++)
        replace_all(x, TTV_ABBREVIATIONS[i][0], TTV_ABBREVIATIONS[i][1]);
    for (i = 0; i < TTV_EXCEPTION_COUNT; i++)
        replace_all(x, TTV_EXCEPTIONS[i][0], TTV_EXCEPTIONS[i][1]);
}

/* Load the working buffer: a leading space, the upper-cased text, a trailing
 * space.  The padding is what lets the context matchers walk one character
 * past the ends without a bounds test. */
static void load_text(ttv_ctx *x, const char *text, int len)
{
    int i;
    int room = TTV_TEXT_MAX - 3;
    if (len > room)
        len = room;
    x->text[0] = ' ';
    for (i = 0; i < len; i++)
        x->text[1 + i] = to_upper(text[i]);
    x->text[1 + len] = ' ';
    x->text[2 + len] = '\0';
    x->text_len = len + 2;
}

/* English spelling to ARPABET.  Upper-case letters spell out two-letter
 * ARPABET symbols; lower-case letters are single-letter consonants.  That
 * convention is the tables', not ours, and the tokenizer below relies on it. */
static void to_arpabet(ttv_ctx *x, const char *text, int len)
{
    int pos = 1, n;

    load_text(x, text, len);
    apply_exceptions(x);
    n = x->text_len;

    x->arpa_len = 0;
    x->arpa_full = 0;

    while (pos < n - 1) {
        char c = x->text[pos];
        const ttv_rule_group *group;
        size_t i;
        int matched = 0;

        if (is_digit(c)) {
            int end = pos;
            while (end < n && is_digit(x->text[end]))
                end++;
            append_number(x, x->text + pos, end - pos);
            pos = end;
            continue;
        }

        group = is_upper(c) ? &TTV_NRL_RULES[1 + (c - 'A')] : &TTV_NRL_RULES[0];

        for (i = 0; i < group->count; i++) {
            const ttv_rule *rule = &group->rules[i];
            int rlen = (int)strlen(rule->match);
            if (strncmp(x->text + pos, rule->match, (size_t)rlen) != 0)
                continue;
            if (!match_left(rule->left, x->text, pos - 1))
                continue;
            if (!match_right(rule->right, x->text, n, pos + rlen))
                continue;
            arpa_puts(x, rule->out);
            pos += rlen;
            matched = 1;
            break;
        }
        if (!matched)
            pos++;   /* no rule: drop the character rather than stall */
    }
    x->arpa[x->arpa_len] = '\0';
}

/* ------------------------------------------------------------- stage two --- */

static int is_map_key(const char *sym)
{
    size_t i;
    for (i = 0; i < TTV_ARPABET_COUNT; i++)
        if (strcmp(sym, TTV_ARPABET[i].arpa) == 0)
            return 1;
    return 0;
}

static int is_two_letter_symbol(const char *s)
{
    char pair[3];
    pair[0] = s[0];
    pair[1] = s[1];
    pair[2] = '\0';
    if (is_map_key(pair))
        return 1;
    /* NG is what the letter-to-sound rules emit for the velar nasal; the map
     * spells the same sound NX.  Without this alias "young" comes out as
     * /n/ + /g/, so treat the two as one symbol and translate below. */
    return s[0] == 'N' && s[1] == 'G';
}

/* The rules write /h/ and /dZ/ as the single letters h and j, but the map keys
 * them as HH and JH.  Nothing else in either table is spelled two ways, so the
 * rule is simply: a lone consonant that is not a key, but becomes one with an
 * H after it, is that symbol.  Without this, /h/ and /dZ/ are silently dropped
 * and "hello" loses its H. */
static void canonical_symbol(char *sym)
{
    char with_h[3];

    if (strcmp(sym, "NG") == 0) {
        memcpy(sym, "NX", 3);
        return;
    }
    if (is_map_key(sym))
        return;
    if (sym[0] != '\0' && sym[1] == '\0') {
        with_h[0] = sym[0];
        with_h[1] = 'H';
        with_h[2] = '\0';
        if (is_map_key(with_h))
            memcpy(sym, with_h, 3);
    }
}

/* Split the ARPABET into symbols.  Two upper-case letters that name a symbol
 * in the map are taken together; anything else is a single character, upper-
 * cased, so the rules' lower-case consonants ("grEYt") land on the map's
 * upper-case keys. */
static void tokenize_arpabet(ttv_ctx *x)
{
    int i = 0;
    x->sym_count = 0;

    while (i < x->arpa_len) {
        char *slot;
        if (x->sym_count >= TTV_SYMS_MAX)
            break;
        slot = x->syms[x->sym_count];

        if (i + 1 < x->arpa_len && is_upper(x->arpa[i]) &&
            is_upper(x->arpa[i + 1]) && is_two_letter_symbol(x->arpa + i)) {
            slot[0] = x->arpa[i];
            slot[1] = x->arpa[i + 1];
            slot[2] = '\0';
            i += 2;
        } else {
            char c = to_upper(x->arpa[i]);
            slot[0] = (c == ' ') ? '_' : c;
            slot[1] = '\0';
            i++;
        }
        canonical_symbol(slot);
        x->sym_count++;
    }
}

int ttv_phone_by_name(const char *name)
{
    int i;
    if (!name)
        return -1;
    for (i = 0; i < 64; i++)
        if (strcmp(name, TTV_PHONE_NAMES[i]) == 0)
            return i;
    return -1;
}

/* ARPABET to SC-01 phone codes.  For each symbol the first map entry whose
 * left and right contexts hold wins; the entries are ordered so that every
 * symbol ends with an unconditional fallback, and lookup cannot fail. */
static void arpabet_to_phones(ttv_ctx *x)
{
    int i;

    tokenize_arpabet(x);
    x->phone_count = 0;

    for (i = 0; i < x->sym_count; i++) {
        const char *left = i ? x->syms[i - 1] : "";
        const char *right = (i + 1 < x->sym_count) ? x->syms[i + 1] : "";
        size_t m;

        for (m = 0; m < TTV_ARPABET_COUNT; m++) {
            const ttv_arpa_map *map = &TTV_ARPABET[m];
            const char *p;

            if (strcmp(x->syms[i], map->arpa) != 0)
                continue;
            if (*map->left && strcmp(left, map->left) != 0)
                continue;
            if (*map->right && strcmp(right, map->right) != 0)
                continue;

            /* `out` is a space-separated list of SC-01 phone names. */
            p = map->out;
            while (*p) {
                const char *end = strchr(p, ' ');
                char name[16];
                size_t len = end ? (size_t)(end - p) : strlen(p);
                int code;
                if (len >= sizeof name)
                    len = sizeof name - 1;
                memcpy(name, p, len);
                name[len] = '\0';
                code = ttv_phone_by_name(name);
                if (code >= 0)
                    phone_put(x, (uint8_t)code);
                if (!end)
                    break;
                p = end + 1;
            }
            break;
        }
    }
}

/* -------------------------------------------------------------- prosody ---
 *
 * The SC-01's pitch input is two bits: four levels, nothing between.  Measured
 * on our own core at the datasheet clock they are 78, 89, 104 and 125 Hz -- a
 * range of about a fifth, in four steps, and that is the entire pitch budget.
 *
 * It is not much, but flat speech is the single most fatiguing thing about
 * early synthesizers, and four levels are enough to carry the one contour that
 * matters most: declination.  English statements drift downward in pitch across
 * a clause and drop at the end; questions do the opposite and rise.  A listener
 * uses that fall to hear where a sentence ends, which is why flat output makes
 * running text feel like it never stops.
 *
 * So: assign a level per phone from its position within its own sentence, and
 * let the sentence's final punctuation choose the shape.  No stress model, no
 * syllable model -- the letter-to-sound rules do not give us either -- but the
 * clause-level contour is the part that four levels can actually express. */

typedef enum { TTV_STATEMENT, TTV_QUESTION, TTV_EXCLAMATION } ttv_sentence;

static ttv_sentence classify(char terminator)
{
    if (terminator == '?') return TTV_QUESTION;
    if (terminator == '!') return TTV_EXCLAMATION;
    return TTV_STATEMENT;
}

static int is_pause_phone(int phone)
{
    return phone == 0x03 || phone == 0x3E;   /* PA0, PA1 */
}

/* The level for a phone `position` of the way through its sentence.
 *
 * Statements start a step above neutral and fall to the floor -- the final drop
 * is what makes a sentence sound finished.  Questions do the reverse, sitting
 * at neutral and climbing over the last third.  Exclamations start high and
 * fall, but stop short of the floor so they keep some energy. */
static int contour_level(ttv_sentence type, double position)
{
    switch (type) {
    case TTV_QUESTION:
        if (position < 0.55) return 1;
        if (position < 0.80) return 2;
        return 3;
    case TTV_EXCLAMATION:
        if (position < 0.20) return 3;
        if (position < 0.55) return 2;
        return 1;
    case TTV_STATEMENT:
    default:
        if (position < 0.25) return 2;
        if (position < 0.70) return 1;
        return 0;
    }
}

/* Stamp a contour across one sentence's phones, in place.
 *
 * Pauses are skipped when measuring position -- they are silent, so counting
 * them would let a comma-heavy sentence spend its whole contour on gaps -- but
 * they still carry the level of whatever preceded them, so the packed stream
 * never has a level discontinuity in it. */
static void apply_contour(uint8_t *phones, size_t count, ttv_sentence type)
{
    size_t voiced = 0, seen = 0, i;
    int level;

    for (i = 0; i < count; i++)
        if (!is_pause_phone(TTV_PHONE_OF(phones[i])))
            voiced++;
    if (voiced == 0)
        return;

    level = contour_level(type, 0.0);
    for (i = 0; i < count; i++) {
        int phone = TTV_PHONE_OF(phones[i]);
        if (!is_pause_phone(phone)) {
            double position = voiced > 1
                ? (double)seen / (double)(voiced - 1)
                : 0.0;
            level = contour_level(type, position);
            seen++;
        }
        phones[i] = TTV_PACK(phone, level);
    }
}

/* ------------------------------------------------------------------- API --- */

/* Copy what fits, report what there was. */
static size_t emit(const ttv_ctx *x, uint8_t *out, size_t capacity)
{
    size_t stored = x->phone_count < TTV_PHONES_MAX
        ? x->phone_count : TTV_PHONES_MAX;
    if (out && capacity > 0) {
        size_t copied = stored < capacity ? stored : capacity;
        memcpy(out, x->phones, copied);
    }
    return x->phone_count;
}

size_t ttv_text_to_phones_flat(const char *text, uint8_t *out, size_t capacity)
{
    ttv_ctx x;
    if (!text)
        return 0;
    to_arpabet(&x, text, (int)strlen(text));
    arpabet_to_phones(&x);
    return emit(&x, out, capacity);
}

/* Split text into sentences at . ? or ! followed by whitespace or the end.
 *
 * The trailing-whitespace requirement is what keeps "3.14" and "1.5" in one
 * piece.  It does not save "e.g." or "Dr." -- an abbreviation ending in a full
 * stop still reads as a sentence end, which costs a contour reset and nothing
 * worse. */
static int is_space(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

size_t ttv_text_to_phones(const char *text, uint8_t *out, size_t capacity)
{
    ttv_ctx x;
    size_t total = 0;
    int len, start = 0, i;

    if (!text)
        return 0;
    len = (int)strlen(text);

    for (i = 0; i <= len; i++) {
        int at_end = (i == len);
        int sent_start = start;
        int sentence_len;
        char terminator = ' ';
        int j;
        size_t n, room;

        if (!at_end) {
            char c = text[i];
            if (c != '.' && c != '?' && c != '!')
                continue;
            /* A terminator at the very end of the text ends a sentence; one in
             * the middle only does if whitespace follows, which is what keeps
             * "3.14" in one piece. */
            if (i + 1 < len && !is_space(text[i + 1]))
                continue;
            sentence_len = i - start + 1;
            start = i + 1;
        } else {
            if (start >= len)
                break;
            sentence_len = len - start;
            start = len;
        }

        to_arpabet(&x, text + sent_start, sentence_len);
        arpabet_to_phones(&x);

        if (x.phone_count == 0)
            continue;

        /* The sentence's shape comes from its last non-space character. */
        for (j = sentence_len - 1; j >= 0; j--) {
            if (!is_space(text[sent_start + j])) {
                terminator = text[sent_start + j];
                break;
            }
        }

        n = x.phone_count < TTV_PHONES_MAX ? x.phone_count : TTV_PHONES_MAX;
        apply_contour(x.phones, n, classify(terminator));

        /* Append what fits; keep counting either way, so the return value is
         * the full length the text produced. */
        if (out && total < capacity) {
            room = capacity - total;
            memcpy(out + total, x.phones, n < room ? n : room);
        }
        total += x.phone_count;
    }
    return total;
}

size_t ttv_spell_text(const char *text, uint8_t *out, size_t capacity)
{
    ttv_ctx x;
    const unsigned char *p;
    size_t i;

    if (!text)
        return 0;

    /* Straight into the ARPABET buffer: spelling skips the letter-to-sound
     * rules entirely, since every character already has a spoken name. */
    x.arpa_len = 0;
    x.arpa_full = 0;
    for (p = (const unsigned char *)text; *p; p++) {
        if (*p > 127)
            continue;
        arpa_puts(&x, TTV_ASCII_NAMES[*p]);
        arpa_putc(&x, ' ');
    }
    x.arpa[x.arpa_len] = '\0';

    arpabet_to_phones(&x);

    for (i = 0; i < x.phone_count && i < TTV_PHONES_MAX; i++)
        x.phones[i] = TTV_PACK(TTV_PHONE_OF(x.phones[i]),
                               TTV_NEUTRAL_INFLECTION);

    return emit(&x, out, capacity);
}
