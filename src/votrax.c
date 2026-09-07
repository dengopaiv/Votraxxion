/* Implementation of the flat C API -- see votrax.h.
 *
 * There is deliberately very little here.  The chip and the front end are
 * whole modules of their own; this file gives them an exported ABI and owns
 * the two pieces of state the API adds: the latched inflection level and the
 * phone scheduler.
 */

#define VOTRAX_BUILD_DLL 1

#include "votrax.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ttv.h"
#include "ttv_tables.h"
#include "votrax_core.h"
#include "votrax_rom.h"

struct vx_chip {
    vx_core core;
    int inflection;

    /* The scheduler.  `queue` is what has been asked for, `hold` is how many
     * samples the phone now sounding still owns.  Rate control is entirely the
     * arithmetic that sets `hold`.
     *
     * A ring buffer rather than a growable queue: this is the one structure on
     * the speech path, and a fixed one means vx_speak cannot allocate, cannot
     * fail for want of memory, and cannot grow without bound if a caller
     * queues faster than the chip drains. */
    unsigned char queue[VX_QUEUE_CAPACITY];
    int head;
    int count;

    double speed;
    int hold;
};

static double clampd(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static vx_mask_revision to_mask(int mask)
{
    return mask == VX_MASK_SC01 ? VX_ROM_SC01 : VX_ROM_SC01A;
}

/* Commit the next queued phone and decide how long to hold it.
 *
 * This is the whole of constant-pitch rate control.  The natural length is
 * exact (16 ticks of 4*duration+1 chip updates, two samples each), so at speed
 * 2.0 the phone is committed at half its natural length and the next one
 * starts there.  The formant interpolators simply carry on toward the new
 * targets from wherever they had reached, which is what keeps fast speech
 * sounding like speech rather than like a chopped-up recording.  Nothing here
 * touches the master clock, so the glottal period -- the pitch -- is
 * untouched. */
static void commit_next(vx_chip *chip)
{
    unsigned char packed = chip->queue[chip->head];
    int phone, level;
    double natural;

    chip->head = (chip->head + 1) % VX_QUEUE_CAPACITY;
    chip->count--;

    phone = TTV_PHONE_OF(packed);
    /* The contour is stored absolutely but applied relative to the base, so a
     * caller who moves the base transposes the whole contour instead of
     * flattening it.  Only four levels exist, so the ends clip. */
    level = chip->inflection + TTV_INFLECTION_OF(packed) - TTV_NEUTRAL_INFLECTION;
    if (level < 0) level = 0;
    if (level > 3) level = 3;

    vx_core_phone_commit(&chip->core, phone, level);

    natural = (double)vx_core_phone_samples(&chip->core, phone);
    chip->hold = (int)lround(natural / chip->speed);
    if (chip->hold < 1)
        chip->hold = 1;
}

/* The front end writes what fits and returns what there was; the API's only
 * job is to keep a negative capacity from becoming an enormous size_t. */
static size_t room(int capacity)
{
    return capacity > 0 ? (size_t)capacity : 0;
}

vx_chip *vx_create(int mask, unsigned int clock_hz)
{
    vx_chip *chip = (vx_chip *)calloc(1, sizeof *chip);
    double clock = clock_hz ? (double)clock_hz : (double)VX_BASE_CLOCK;
    if (!chip)
        return NULL;
    chip->inflection = VX_NEUTRAL_INFLECTION;
    chip->speed = 1.0;
    vx_core_init(&chip->core, clock, 150.0 / 4000.0, 1.0, 1.0, 1.0,
                 to_mask(mask));
    return chip;
}

void vx_destroy(vx_chip *chip)
{
    free(chip);
}

void vx_reset(vx_chip *chip)
{
    if (!chip) return;
    vx_core_reset(&chip->core);
    chip->inflection = VX_NEUTRAL_INFLECTION;
    chip->head = 0;
    chip->count = 0;
    chip->hold = 0;
}

void vx_set_clock(vx_chip *chip, unsigned int hz)
{
    if (!chip || !hz) return;
    /* The core takes its clock at init, so re-init in place, keeping the mask.
     * Callers change rate between utterances, not mid-phone. */
    vx_core_init(&chip->core, (double)hz, 150.0 / 4000.0, 1.0, 1.0, 1.0,
                 chip->core.mask);
    chip->hold = 0;   /* hold times are in samples; the sample rate just moved */
}

unsigned int vx_clock(vx_chip *chip)
{
    return chip ? (unsigned int)lround(chip->core.master_clock) : 0u;
}

double vx_sample_rate(vx_chip *chip)
{
    return chip ? chip->core.sclock : 0.0;
}

void vx_set_mask(vx_chip *chip, int mask)
{
    if (chip) vx_core_set_mask(&chip->core, to_mask(mask));
}

int vx_mask(vx_chip *chip)
{
    if (!chip) return VX_MASK_SC01A;
    return chip->core.mask == VX_ROM_SC01 ? VX_MASK_SC01 : VX_MASK_SC01A;
}

void vx_inflection(vx_chip *chip, unsigned char level)
{
    if (chip) chip->inflection = level & 0x03;
}

int vx_get_inflection(vx_chip *chip)
{
    return chip ? chip->inflection : VX_NEUTRAL_INFLECTION;
}

void vx_set_speed(vx_chip *chip, double speed)
{
    if (!chip) return;
    chip->speed = clampd(speed, 0.1, 10.0);
}

double vx_speed(vx_chip *chip)
{
    return chip ? chip->speed : 1.0;
}

int vx_phone_samples(vx_chip *chip, unsigned char phone)
{
    return chip ? vx_core_phone_samples(&chip->core, phone & 0x3F) : 0;
}

void vx_write(vx_chip *chip, unsigned char phone)
{
    if (!chip) return;
    vx_core_phone_commit(&chip->core, phone & 0x3F, chip->inflection);
    chip->hold = 0;   /* hand control back to the scheduler cleanly */
}

int vx_speak(vx_chip *chip, const unsigned char *phones, int count)
{
    int i;
    if (!chip) return 0;
    if (!phones || count <= 0) return chip->count;
    for (i = 0; i < count && chip->count < VX_QUEUE_CAPACITY; i++) {
        int tail = (chip->head + chip->count) % VX_QUEUE_CAPACITY;
        chip->queue[tail] = phones[i];
        chip->count++;
    }
    return chip->count;
}

int vx_pending(vx_chip *chip)
{
    return chip ? chip->count : 0;
}

void vx_cancel(vx_chip *chip)
{
    int level;
    if (!chip) return;
    chip->head = 0;
    chip->count = 0;
    chip->hold = 0;
    /* Reset, then put back the inflection the reset cleared: a cancel must not
     * silently change the voice for whatever is spoken next. */
    level = chip->inflection;
    vx_core_reset(&chip->core);
    chip->inflection = level;
}

int vx_ready(vx_chip *chip)
{
    return (chip && vx_core_phone_done(&chip->core)) ? 1 : 0;
}

int vx_render(vx_chip *chip, int16_t *buffer, int count)
{
    int i;
    if (!chip || !buffer || count <= 0) return 0;
    for (i = 0; i < count; i++) {
        double s;
        /* Commit the next phone when the current one's hold expires.  At speed
         * > 1 that happens before the chip would have asked for it, which is
         * the truncation; at speed < 1 it happens after, and the chip sustains
         * the phone in the meantime.
         *
         * Commit before generating and decrement after, so a phone gets
         * exactly `hold` samples.  Decrementing first costs it one. */
        if (chip->hold == 0 && chip->count > 0)
            commit_next(chip);

        s = clampd(vx_core_generate_one_sample(&chip->core), -1.0, 1.0);
        buffer[i] = (int16_t)lround(s * 32767.0);

        if (chip->hold > 0)
            chip->hold--;
    }
    return count;
}

/* --- text to phones ------------------------------------------------------ */

int ttv_translate(const char *text, unsigned char *out, int capacity)
{
    if (!text) return 0;
    return (int)ttv_text_to_phones(text, out, room(capacity));
}

int ttv_translate_flat(const char *text, unsigned char *out, int capacity)
{
    if (!text) return 0;
    return (int)ttv_text_to_phones_flat(text, out, room(capacity));
}

int ttv_spell(const char *text, unsigned char *out, int capacity)
{
    if (!text) return 0;
    return (int)ttv_spell_text(text, out, room(capacity));
}

const char *vx_phone_name(int code)
{
    return ttv_phone_name(code);
}

int vx_phone_by_name(const char *name)
{
    return name ? ttv_phone_by_name(name) : -1;
}
