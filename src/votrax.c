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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* One first-order section, bilinear with the corner prewarped. */
typedef struct {
    double b0, b1, a1;
    double x1, y1;
} vx_pole;

struct vx_chip {
    vx_core core;
    int inflection;
    /* The contour step of the phone now sounding, relative to neutral, so a
     * pitch change mid-phone can be applied at once without losing it. */
    int contour;

    /* The output stage (VX_OUTPUT_*): three first-order sections, rebuilt
     * whenever the sample rate moves.  See build_output_stage. */
    int output;
    vx_pole out_hp_in, out_lp, out_hp_spk;

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

static int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* --- the data sheet's output stage ----------------------------------------
 *
 * Figure 8 on page 10 of the data sheet (the SC-01A "typical application") is
 * the one complete audio circuit Votrax published.  Read off the scan, with
 * the volume control at full:
 *
 *   AO (pin 22) -- 1 uF -- 4.7 k --+-- 0.05 uF to ground
 *                                  +-- 10 k audio-taper pot to ground;
 *                                      wiper at the top, so also:
 *                                  +-- 1.2 k to ground     (LM386 pin 3)
 *                                  +-- 0.05 uF to ground   (pin 3 to pin 2)
 *   LM386N-1, gain 20 (pins 1 and 8 open) -- 330 uF -- 8 ohm speaker
 *
 * The 6.8 k resistor that the 1 uF line appears to touch belongs to the clock
 * network; the scan draws a crossover hop there, not a junction.
 *
 * That is three first-order sections:
 *
 *   input coupling   1 uF into 4.7 k + (10 k || 1.2 k)         high-pass  27.6 Hz
 *   RC network       (4.7 k || 10 k || 1.2 k) against 0.1 uF    low-pass   1825 Hz
 *   output coupling  330 uF into the 8 ohm speaker              high-pass  60.3 Hz
 *
 * The passband gain (0.186 from the divider, 20 from the LM386) is left out:
 * the level stays the chip's, so switching the stage changes the tone rather
 * than the loudness of the band below the corner.  The corners are set by the
 * parts and do not move with the master clock -- the voice changes and the
 * loudspeaker circuit does not, which is how the real board behaved.  Not
 * modelled: the LM386's own limits, the speaker's response, and the volume
 * control at anything but full. */

#define VX_FIG8_HP_IN_HZ \
    (1.0 / (2.0 * M_PI * 1e-6 * (4700.0 + 10000.0 * 1200.0 / 11200.0)))
#define VX_FIG8_LP_HZ \
    (1.0 / (2.0 * M_PI * 0.1e-6 * \
            (1.0 / (1.0 / 4700.0 + 1.0 / 10000.0 + 1.0 / 1200.0))))
#define VX_FIG8_HP_SPK_HZ (1.0 / (2.0 * M_PI * 330e-6 * 8.0))

static void pole_build(vx_pole *p, double corner_hz, double rate, int highpass)
{
    double corner = corner_hz < rate * 0.45 ? corner_hz : rate * 0.45;
    double k = tan(M_PI * corner / rate);
    if (highpass) {
        p->b0 = 1.0 / (1.0 + k);
        p->b1 = -p->b0;
    } else {
        p->b0 = k / (1.0 + k);
        p->b1 = p->b0;
    }
    p->a1 = (k - 1.0) / (k + 1.0);
}

static double pole_run(vx_pole *p, double x)
{
    double y = p->b0 * x + p->b1 * p->x1 - p->a1 * p->y1;
    p->x1 = x;
    p->y1 = y;
    return y;
}

static void pole_clear(vx_pole *p)
{
    p->x1 = p->y1 = 0.0;
}

static void build_output_stage(vx_chip *chip)
{
    double rate = chip->core.sclock;
    pole_build(&chip->out_hp_in, VX_FIG8_HP_IN_HZ, rate, 1);
    pole_build(&chip->out_lp, VX_FIG8_LP_HZ, rate, 0);
    pole_build(&chip->out_hp_spk, VX_FIG8_HP_SPK_HZ, rate, 1);
}

static void clear_output_stage(vx_chip *chip)
{
    pole_clear(&chip->out_hp_in);
    pole_clear(&chip->out_lp);
    pole_clear(&chip->out_hp_spk);
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
    chip->contour = TTV_INFLECTION_OF(packed) - TTV_NEUTRAL_INFLECTION;
    level = clampi(chip->inflection + chip->contour, 0, 3);

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
    chip->output = VX_OUTPUT_CHIP;
    vx_core_init(&chip->core, clock, 150.0 / 4000.0, 1.0, 1.0, 1.0,
                 to_mask(mask));
    build_output_stage(chip);
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
    chip->contour = 0;
    chip->head = 0;
    chip->count = 0;
    chip->hold = 0;
    clear_output_stage(chip);
}

void vx_set_clock(vx_chip *chip, unsigned int hz)
{
    if (!chip || !hz) return;
    /* A live change, as the data sheet's potentiometer and DAC figures make:
     * the chip keeps talking.  Hold times need no rescaling -- a phone is a
     * fixed number of samples at any clock (vx_phone_samples) -- but the
     * output stage's corners are in hertz and are rebuilt for the new rate. */
    vx_core_set_clock(&chip->core, (double)hz);
    build_output_stage(chip);
}

unsigned int vx_clock_from_rc(double ohms, double farads)
{
    double hz;
    if (!(ohms > 0.0) || !(farads > 0.0))
        return 0u;
    hz = 1.25 / (ohms * farads);
    return hz >= 4294967295.0 ? 4294967295u : (unsigned int)lround(hz);
}

unsigned int vx_clock_from_knob(double position)
{
    /* Audio (logarithmic) taper by the conventional 40 dB law: 10% of the
     * track at mid rotation.  The data sheet names the taper, not the law. */
    double p = clampd(position, 0.0, 1.0);
    double fraction = (pow(10.0, 2.0 * p) - 1.0) / 99.0;
    return vx_clock_from_rc(VX_KNOB_FIXED_OHMS + VX_KNOB_POT_OHMS * fraction,
                            VX_KNOB_FARADS);
}

void vx_set_output(vx_chip *chip, int stage)
{
    if (!chip) return;
    stage = stage == VX_OUTPUT_FIGURE8 ? VX_OUTPUT_FIGURE8 : VX_OUTPUT_CHIP;
    if (stage != chip->output)
        clear_output_stage(chip);
    chip->output = stage;
}

int vx_output(vx_chip *chip)
{
    return chip ? chip->output : VX_OUTPUT_CHIP;
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
    if (!chip) return;
    chip->inflection = level & 0x03;
    /* Immediately, as the I1/I2 pins do: the phone already sounding moves to
     * the new level, keeping its place in the sentence contour. */
    vx_core_set_inflection(&chip->core,
                           clampi(chip->inflection + chip->contour, 0, 3));
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
    chip->contour = 0;
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
    chip->contour = 0;
    clear_output_stage(chip);
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

        s = vx_core_generate_one_sample(&chip->core);
        if (chip->output == VX_OUTPUT_FIGURE8)
            s = pole_run(&chip->out_hp_spk,
                         pole_run(&chip->out_lp,
                                  pole_run(&chip->out_hp_in, s)));
        s = clampd(s, -1.0, 1.0);
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
