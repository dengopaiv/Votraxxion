/* Flat C API over the SC-01 synthesizer.
 *
 * This is the surface an NVDA add-on binds to: a handful of entry points that
 * ctypes can call without a C++ ABI, a compiler, or any knowledge of what is
 * behind them.  Everything the synthesizer needs is compiled in -- both mask
 * ROMs and the whole English front end -- so the add-on ships one library file
 * and nothing else.
 *
 * Threading: a vx_chip is not internally locked.  The intended shape, and the
 * one Tamas Geczy's votraxsc01 add-on uses (the driver in nvda-addon/ descends
 * from it -- see NOTICE.md), is that exactly one thread touches the chip
 * after construction -- a speak thread that writes phones and pulls audio --
 * while the caller's thread only queues work for it.  ttv_translate and
 * ttv_spell touch no chip state at all and are reentrant.
 */
#ifndef VOTRAX_H
#define VOTRAX_H

#include <stdint.h>

/* VOTRAX_STATIC: the sources are compiled straight into the program, so there
 * is no import and nothing to export.  gui-native/ is built that way -- the
 * whole synthesizer is six .c files and there is no reason for a one-window
 * program to carry a DLL beside it. */
#if defined(VOTRAX_STATIC)
#  define VOTRAX_API
#elif defined(_WIN32)
#  if defined(VOTRAX_BUILD_DLL)
#    define VOTRAX_API __declspec(dllexport)
#  else
#    define VOTRAX_API __declspec(dllimport)
#  endif
#else
#  define VOTRAX_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vx_chip vx_chip;

/* Mask revisions, matching vx_mask_revision in votrax_rom.h. */
#define VX_MASK_SC01A 0   /* the later revision (default) */
#define VX_MASK_SC01  1   /* the 1980 part: louder, more strident open vowels */

/* The datasheet master clock.  Varying it changes speed and pitch together,
 * the way the 1980 hardware's one knob did. */
#define VX_BASE_CLOCK 720000u

/* A phone and its inflection travel together in one byte: the phone in bits
 * 0-5, the level in bits 6-7.  A byte with its top bits clear is just a plain
 * phone code, so packed and unpacked streams are interchangeable. */
#define VX_PHONE(b)          ((unsigned char)((b) & 0x3F))
#define VX_INFLECTION(b)     ((unsigned char)(((b) >> 6) & 0x03))
#define VX_PACK(phone, lvl)  ((unsigned char)(((phone) & 0x3F) | (((lvl) & 3) << 6)))

/* The level a contour treats as neutral: set the base here and the contour
 * from ttv_translate comes through exactly as computed. */
#define VX_NEUTRAL_INFLECTION 1

/* --- chip lifecycle ------------------------------------------------------ */

/* Create a chip on one of the two mask ROMs.  `clock_hz` of 0 means
 * VX_BASE_CLOCK.  Returns NULL only on allocation failure -- there is no ROM
 * file to find and therefore no way for this to fail on bad data. */
VOTRAX_API vx_chip *vx_create(int mask, unsigned int clock_hz);
VOTRAX_API void vx_destroy(vx_chip *chip);

/* Instant silence: clears the interpolation state and re-latches STOP.
 * Preserves the clock and the mask; zeroes inflection. */
VOTRAX_API void vx_reset(vx_chip *chip);

/* --- configuration ------------------------------------------------------- */

VOTRAX_API void vx_set_clock(vx_chip *chip, unsigned int hz);
VOTRAX_API unsigned int vx_clock(vx_chip *chip);

/* Output sample rate in Hz -- master clock / 18.  Changes with the clock, so
 * re-read it after vx_set_clock and re-open the audio device if it moved. */
VOTRAX_API double vx_sample_rate(vx_chip *chip);

VOTRAX_API void vx_set_mask(vx_chip *chip, int mask);
VOTRAX_API int vx_mask(vx_chip *chip);

/* The chip's 2-bit pitch input: four levels, 0-3, nothing between.  At the
 * datasheet clock they are about 78, 89, 104 and 125 Hz.
 *
 * This is the base level, and it means two slightly different things depending
 * on how the chip is driven.  For vx_write it is simply the level used.  For
 * the scheduler it is the level a neutral phone gets, and a contour shifts
 * around it: the final level is base + packed - VX_NEUTRAL_INFLECTION, clamped
 * to 0-3.  So leaving it at the default of 1 reproduces ttv_translate's
 * contour exactly, and moving it transposes the whole contour.
 *
 * Note the consequence of having only four levels: at base 0 or 3 the contour
 * is clipped flat against the end of the range.  That is the hardware's limit,
 * not a bug, and it is why a caller mapping a 0-100 pitch setting onto this
 * should favour the middle. */
VOTRAX_API void vx_inflection(vx_chip *chip, unsigned char level);
VOTRAX_API int vx_get_inflection(vx_chip *chip);

/* --- rate ----------------------------------------------------------------
 *
 * Two ways to change speed, and they are not the same.
 *
 * vx_set_clock is the 1980 hardware's single knob: it moves the master clock,
 * so tempo and pitch rise together and the voice turns into a chipmunk.  It is
 * the authentic behaviour and worth offering, but it is not what a screen
 * reader wants at 400 words a minute.
 *
 * vx_set_speed leaves the clock alone and truncates instead: each phone is
 * held for its natural length divided by `speed`, and the next is committed
 * early.  The duration counter is the only thing that moves; the glottal
 * oscillator never sees it.  Tempo changes, pitch does not, and the audio is
 * still the chip's own output -- no resampling, no time-stretch artefacts.
 * Speeds below 1.0 extend instead, sustaining each phone past its natural end.
 *
 * Rate by phone truncation, with the clock kept as an "authentic rate" option,
 * is Geczy's design from his votraxsc01 NVDA driver, where it lived in Python
 * and measured each phone at startup.  Here it is the scheduler, working from
 * the exact phone length (vx_phone_samples).
 *
 * The two compose: set a clock for the voice you want, then a speed for the
 * tempo you want. */

/* 1.0 is natural; >1 faster, <1 slower.  Clamped to [0.1, 10].  Applies from
 * the next phone the scheduler commits. */
VOTRAX_API void vx_set_speed(vx_chip *chip, double speed);
VOTRAX_API double vx_speed(vx_chip *chip);

/* Natural length of a phone in samples at the current clock -- what the
 * scheduler divides by `speed`.  Exact, not measured. */
VOTRAX_API int vx_phone_samples(vx_chip *chip, unsigned char phone);

/* --- speaking ------------------------------------------------------------
 *
 * Two ways to drive the chip, and they can be mixed.
 *
 * The low-level way is vx_write plus vx_ready: hand it a phone whenever it
 * asks for one.  That is the hardware's own protocol and gives the caller
 * complete control.
 *
 * The high-level way is vx_speak plus vx_render: queue phones and pull audio.
 * The scheduler owns the timing, which is where rate control lives, so this is
 * the path a screen reader wants. */

/* How many phones the queue holds.  A screen reader's longest utterance is far
 * below this; a caller with more should speak it in pieces, which it wants to
 * do anyway so that cancelling stays responsive. */
#define VX_QUEUE_CAPACITY 1024

/* Latch a phone (0-63) and start voicing it, ignoring the queue.  Writing
 * before vx_ready truncates the phone in progress. */
VOTRAX_API void vx_write(vx_chip *chip, unsigned char phone);

/* Non-zero once the current phone has finished and the chip wants the next. */
VOTRAX_API int vx_ready(vx_chip *chip);

/* Queue phones for the scheduler.  Returns the number now queued.  Phones
 * beyond VX_QUEUE_CAPACITY are dropped rather than growing the queue, so the
 * return value is worth checking if you are pushing a lot at once. */
VOTRAX_API int vx_speak(vx_chip *chip, const unsigned char *phones, int count);

/* How many phones are still queued.  Zero does not mean silent: the chip is
 * still voicing the last one it was given. */
VOTRAX_API int vx_pending(vx_chip *chip);

/* Cancel: drop the queue and silence the chip immediately, keeping the clock,
 * mask, speed and inflection.
 *
 * Dropping the queue alone is not enough, and this is the one piece of the
 * protocol that is easy to get wrong.  The chip latches a phone and voices it
 * to completion, so after a cancel it keeps producing the last phone it was
 * given.  If the next utterance simply renders on from there, that phone's
 * remainder comes out first and is heard as a scrap of the cancelled speech at
 * the head of the new one.  Resetting clears the interpolation state.
 *
 * Geczy diagnosed and fixed this in votraxsc01 1.0.2 (users heard it when
 * tabbing quickly); vx_cancel is the same fix moved into the library. */
VOTRAX_API void vx_cancel(vx_chip *chip);

/* Render up to `count` mono 16-bit samples, committing queued phones as their
 * hold times expire.  Returns the number of samples written, always `count`.
 * With nothing queued it simply renders whatever the chip is voicing, so it is
 * safe to keep calling for a tail after the last phone. */
VOTRAX_API int vx_render(vx_chip *chip, int16_t *buffer, int count);

/* --- text to phones ------------------------------------------------------ */

/* Translate English text to packed phone bytes, written into `out`.  Returns
 * the number of phones, or the number that would have been produced if it
 * exceeds `capacity` (so a short buffer is detectable).
 *
 * Each sentence gets a pitch contour: statements fall, questions rise.  Use
 * VX_PHONE and VX_INFLECTION to unpack, or hand the bytes to vx_speak, which
 * understands them. */
VOTRAX_API int ttv_translate(const char *text, unsigned char *out, int capacity);

/* The same, flat -- every phone at inflection 0, for a caller supplying its
 * own contour. */
VOTRAX_API int ttv_translate_flat(const char *text, unsigned char *out,
                                  int capacity);

/* Read the text out character by character.  Deliberately flat: a spelled-out
 * string is a list, not a sentence. */
VOTRAX_API int ttv_spell(const char *text, unsigned char *out, int capacity);

/* --- phone names --------------------------------------------------------- */

/* Datasheet name of a phone code, or NULL if out of range.  Valid for the
 * lifetime of the library; do not free. */
VOTRAX_API const char *vx_phone_name(int code);

/* Phone code for a datasheet name (case sensitive), or -1 if unknown. */
VOTRAX_API int vx_phone_by_name(const char *name);

#ifdef __cplusplus
}
#endif

#endif /* VOTRAX_H */
