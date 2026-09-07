/* Text to SC-01 phone codes -- the English front end.
 *
 * Two stages, walking the tables in ttv_tables.c:
 *
 *   text  --TTV_NRL_RULES-->  ARPABET  --TTV_ARPABET-->  phone codes
 *
 * Nothing here allocates, reads a file, or needs a dictionary: nine hundred-odd
 * table entries and about three hundred lines of matcher are the whole thing.
 * That is the point -- an NVDA add-on gets a native library it can drive over a
 * few C calls, with no Python and no data files to install alongside it.
 *
 * See docs/tech-overview.md, Part 4, for the rule format and table provenance.
 *
 * Bounds.  The C++ this replaced grew std::strings and had no limits; these
 * functions work in fixed buffers sized well above anything a screen reader
 * hands over in one call (NVDA speaks a line at a time).  Longer input is
 * truncated rather than growing the buffers, and the sizes are public so a
 * caller that means to feed it a paragraph can split first.
 */
#ifndef TTV_H
#define TTV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Working-buffer sizes.  Input beyond TTV_TEXT_MAX is truncated at a character
 * boundary before any rule runs, so the result is what the truncated text
 * would have produced -- not a partial anything.  Beyond that, an utterance
 * whose ARPABET or phone stream overflows is truncated at a symbol boundary,
 * and the returned count still reports the full length, so a caller can tell. */
#define TTV_TEXT_MAX   4096   /* uppercased, space-padded working text        */
#define TTV_ARPA_MAX   8192   /* the ARPABET between the two stages (~2.4x)   */
#define TTV_SYMS_MAX   4096   /* tokenized ARPABET symbols                    */
#define TTV_PHONES_MAX 4096   /* phone codes for one call                     */

/* A phone code and its inflection travel together in one byte: the phone in
 * bits 0-5, the level in bits 6-7.  A byte whose top bits are clear is
 * therefore just a plain phone code, which is what makes the packed and
 * unpacked forms interchangeable everywhere a phone is accepted. */
#define TTV_PHONE_MASK      0x3F
#define TTV_INFLECTION_SHIFT 6

/* The level a contour treats as neutral.  Levels are stored absolutely, but a
 * caller's own pitch setting shifts the whole contour relative to this, so a
 * caller who sets "normal" gets exactly the contour computed here. */
#define TTV_NEUTRAL_INFLECTION 1

/* The phone the chip idles on, and what closes an utterance. */
#define TTV_STOP 0x3F
/* A word gap. */
#define TTV_PA0  0x03

#define TTV_PACK(phone, level) \
    ((uint8_t)(((phone) & TTV_PHONE_MASK) | (((level) & 3) << TTV_INFLECTION_SHIFT)))
#define TTV_PHONE_OF(b)      ((int)((b) & TTV_PHONE_MASK))
#define TTV_INFLECTION_OF(b) ((int)(((b) >> TTV_INFLECTION_SHIFT) & 3))

/* Translate English text to packed phone bytes, with a pitch contour over each
 * sentence.  Writes at most `capacity` bytes into `out` and returns the number
 * of phones the text produced, which may exceed `capacity`. */
size_t ttv_text_to_phones(const char *text, uint8_t *out, size_t capacity);

/* The same, flat: every phone at inflection 0, for a caller supplying its own
 * contour. */
size_t ttv_text_to_phones_flat(const char *text, uint8_t *out, size_t capacity);

/* Read text out character by character, the way a screen reader reads a
 * password field or an unfamiliar word.  Control codes get their names too.
 * Deliberately flat: a spelled-out string is a list, not a sentence, and a
 * declination contour over it would imply a shape it does not have. */
size_t ttv_spell_text(const char *text, uint8_t *out, size_t capacity);

/* Phone code for a datasheet name (case sensitive), or -1 if unknown. */
int ttv_phone_by_name(const char *name);

#ifdef __cplusplus
}
#endif

#endif /* TTV_H */
