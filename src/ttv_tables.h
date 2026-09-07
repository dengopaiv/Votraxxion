/* Letter-to-sound tables -- declarations.  The data, and the long note on
 * where it came from and how the rule format works, are in ttv_tables.c. */
#ifndef TTV_TABLES_H
#define TTV_TABLES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One NRL letter-to-sound rule: at the cursor, if `match` is present and both
 * contexts hold, emit `out` and advance by strlen(match).  `left` is written
 * in reverse reading order, so the character nearest the cursor comes last. */
typedef struct {
    const char *left;
    const char *match;
    const char *right;
    const char *out;
} ttv_rule;

/* The rules for one leading character, in priority order. */
typedef struct {
    const ttv_rule *rules;
    size_t count;
} ttv_rule_group;

/* One ARPABET-to-SC-01 mapping.  `left` and `right` are the adjacent ARPABET
 * symbols, an empty string meaning "any"; `out` is a space-separated list of
 * SC-01 phone names. */
typedef struct {
    const char *left;
    const char *arpa;
    const char *right;
    const char *out;
} ttv_arpa_map;

/* Indexed 0 for punctuation, then 1 + (letter - 'A'). */
extern const ttv_rule_group TTV_NRL_RULES[27];

extern const ttv_arpa_map TTV_ARPABET[];
extern const size_t TTV_ARPABET_COUNT;

/* Whole-word rewrites applied before the rules run, as {as written, as
 * respelled}.  Both are space-padded so the blanks act as word boundaries.
 * Abbreviations are expanded first, so the expansion is then subject to the
 * ordinary rules. */
extern const char *const TTV_ABBREVIATIONS[][2];
extern const size_t TTV_ABBREVIATION_COUNT;
extern const char *const TTV_EXCEPTIONS[][2];
extern const size_t TTV_EXCEPTION_COUNT;

/* Number names as ARPABET.  [0..19] are zero..nineteen, [20..27] twenty,
 * thirty..ninety. */
extern const char *const TTV_CARDINALS[28];
extern const char *const TTV_ORDINALS[28];

/* Spoken names for the 128 ASCII codes, indexed by the character's own code. */
extern const char *const TTV_ASCII_NAMES[128];

/* The 64 SC-01 phone names, indexed by phone code. */
extern const char *const TTV_PHONE_NAMES[64];

/* Datasheet name of a phone code ("EH3", "PA0", "STOP"...), or NULL if the
 * code is out of range.  Valid for the life of the program; do not free. */
const char *ttv_phone_name(int code);

/* Staged for a currency and ordinal reader the front end does not have yet.
 * Kept with the tables so that whoever writes it does not have to invent the
 * pronunciations again. */
extern const char *const TTV_POINT;
extern const char *const TTV_DOLLAR;
extern const char *const TTV_DOLLARS;
extern const char *const TTV_AND;
extern const char *const TTV_CENT;
extern const char *const TTV_CENTS;

#ifdef __cplusplus
}
#endif

#endif /* TTV_TABLES_H */
