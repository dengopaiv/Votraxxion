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

/* One second-order section, direct form I. */
typedef struct {
    double b[3], a[3];
    double x1, x2, y1, y2;
} vx_biquad;

struct vx_chip {
    vx_core core;
    int inflection;
    /* The contour step of the phone now sounding, relative to neutral, so a
     * pitch change mid-phone can be applied at once without losing it. */
    int contour;

    /* The output stage (VX_OUTPUT_*) and its volume control, rebuilt whenever
     * the sample rate or the volume moves.  See build_output_stage. */
    int output;
    double volume;
    double out_r_in;
    vx_pole out_hp_in, out_hp_spk;
    vx_biquad out_net;

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

/* --- the data sheets' output stage ---------------------------------------
 *
 * Figure 8 on page 10 of the SC-01 data sheet (the SC-01A "typical
 * application") is the one complete audio circuit Votrax published.  Read off
 * the scan at 300 dpi:
 *
 *   AO (pin 22) -- 1 uF -- R1 4.7 k --A-- C1 0.05 uF to ground
 *                                     |
 *                                     10 k audio-taper volume: A to wiper Rt,
 *                                     wiper to ground Rb, Rt + Rb = 10 k
 *                                     |
 *                                     W-- 1.2 k to ground
 *                                      -- C2 0.05 uF to ground (pin 3 to pin 2)
 *                                      -- LM386N-1 pin 3
 *   LM386N-1, pins 1 and 8 open -- 330 uF -- 8 ohm speaker        Vp = 12 V
 *
 * The 6.8 k resistor the 1 uF line seems to touch belongs to the clock; the
 * scan draws a crossover hop there, not a junction.
 *
 * The amplifier is taken from TI's LM386 data sheet (SNAS545D): input
 * resistance 50 k (pin 3 to ground, in parallel with the 1.2 k); voltage gain
 * 20 with pins 1 and 8 open; bandwidth 300 kHz, which is -0.02 dB at 20 kHz
 * and is not modelled; and an output that clips hard -- Figure 6-6 shows
 * distortion flat at about 0.2% and then vertical -- at 6.6 V peak to peak
 * into 8 ohms on a 12 V supply (Figure 6-3).
 *
 * Levels need the SC-01 sheet too: AO swings 0.18-0.26 x Vp peak to peak on
 * AH.  Taking the middle, 0.22 x 12 V = 2.64 V, against AH on the 1980 mask at
 * the neutral level -- 1.383 chip units peak to peak -- gives 1.909 V per chip
 * unit.  Both masks share that scale; the later mask's quieter vowels are
 * quieter in volts too.
 *
 * So, in order:
 *
 *   input coupling  1 uF into R1 + Rt + (Rb || 1.2 k || 50 k)   high-pass
 *   the network     R1, C1, Rt, Rb || 1.2 k || 50 k, C2        second order
 *   LM386           x 20, hard limit at +-3.3 V
 *   output coupling 330 uF into 8 ohms                          high-pass 60.3 Hz
 *
 * At full volume Rt is zero, C1 and C2 are one 0.1 uF, and the network is a
 * single pole at 1855 Hz with a gain of 0.1825.  The output is scaled back by
 * that full-volume gain, 20 and the volts per unit, so an unclipped signal at
 * full volume comes out at the chip's own level; turning the volume down makes
 * it quieter, as the knob did.
 *
 * The finding this produces: at full volume the LM386 runs out of swing when
 * low-frequency AO drive passes +-0.47 chip units, 0.95 peak to peak, and AH
 * on the 1980 mask is 1.38 units peak to peak -- Votrax's own reference
 * circuit clips its loudest vowels unless the volume is turned down.  The
 * later mask's AH, 0.83, fits.
 *
 * Not modelled: the LM386's 0.2% distortion below clipping, the speaker, and
 * AO's source resistance (90 ohms at most, beside 4.7 k). */

#define VX_FIG8_VP              12.0
#define VX_FIG8_VOLTS_PER_UNIT  (0.22 * VX_FIG8_VP / 1.383)
#define VX_FIG8_R1              4700.0
#define VX_FIG8_POT             10000.0
#define VX_FIG8_R_WIPER         1200.0
#define VX_LM386_R_IN           50000.0     /* SNAS545D 6.5, RIN          */
#define VX_LM386_GAIN           20.0        /* SNAS545D 6.5, AV, 1-8 open */
#define VX_LM386_SWING          3.3         /* Figure 6-3, 12 V, 8 ohms   */
#define VX_FIG8_C1              0.05e-6
#define VX_FIG8_C2              0.05e-6
#define VX_FIG8_C_IN            1e-6
#define VX_FIG8_C_OUT           330e-6
#define VX_FIG8_SPEAKER         8.0

/* The conventional audio taper: 10% of the track at half rotation.  Both data
 * sheets name the taper and neither gives its law. */
static double audio_taper(double position)
{
    double p = clampd(position, 0.0, 1.0);
    return (pow(10.0, 2.0 * p) - 1.0) / 99.0;
}

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

static double biquad_run(vx_biquad *q, double x)
{
    double y = q->b[0] * x + q->b[1] * q->x1 + q->b[2] * q->x2
             - q->a[1] * q->y1 - q->a[2] * q->y2;
    q->x2 = q->x1;
    q->x1 = x;
    q->y2 = q->y1;
    q->y1 = y;
    return y;
}

static void biquad_clear(vx_biquad *q)
{
    q->x1 = q->x2 = q->y1 = q->y2 = 0.0;
}

/* The network from AO's coupling capacitor to LM386 pin 3, by nodal analysis
 * at A and W:
 *
 *   H(s) = G1 / (G1 + Gw + Rt G1 Gw + s((1 + Rt Gw) C1 + (1 + Rt G1) C2)
 *                + s^2 Rt C1 C2)
 *
 * with G1 = 1/R1 and Gw = 1/Rb + 1/1.2 k + 1/50 k.  At Rt = 0 it is the single
 * pole of the full-volume reading.  Bilinear, prewarped at the full-volume
 * corner so the audible corner lands where the parts put it. */
static void build_network(vx_chip *chip, double rate)
{
    vx_biquad *q = &chip->out_net;
    double rb = VX_FIG8_POT * audio_taper(chip->volume);
    double rt = VX_FIG8_POT - rb;
    double g1 = 1.0 / VX_FIG8_R1;
    double gw, a0, a1, a2, c, d0, corner;

    if (rb <= 0.0) {                         /* volume at zero: silence */
        q->b[0] = q->b[1] = q->b[2] = 0.0;
        q->a[0] = 1.0;
        q->a[1] = q->a[2] = 0.0;
        chip->out_r_in = VX_FIG8_R1 + VX_FIG8_POT;
        return;
    }
    gw = 1.0 / rb + 1.0 / VX_FIG8_R_WIPER + 1.0 / VX_LM386_R_IN;
    a0 = g1 + gw + rt * g1 * gw;
    a1 = (1.0 + rt * gw) * VX_FIG8_C1 + (1.0 + rt * g1) * VX_FIG8_C2;
    a2 = rt * VX_FIG8_C1 * VX_FIG8_C2;

    corner = (g1 + 1.0 / VX_FIG8_POT + 1.0 / VX_FIG8_R_WIPER + 1.0 / VX_LM386_R_IN)
           / (2.0 * M_PI * (VX_FIG8_C1 + VX_FIG8_C2));
    if (corner > rate * 0.45)
        corner = rate * 0.45;
    c = 2.0 * M_PI * corner / tan(M_PI * corner / rate);

    d0 = a0 + a1 * c + a2 * c * c;
    q->b[0] = g1 / d0;
    q->b[1] = 2.0 * g1 / d0;
    q->b[2] = g1 / d0;
    q->a[0] = 1.0;
    q->a[1] = (2.0 * a0 - 2.0 * a2 * c * c) / d0;
    q->a[2] = (a0 - a1 * c + a2 * c * c) / d0;

    /* What the input coupling capacitor sees at DC: R1, the upper track and
     * everything from the wiper to ground. */
    chip->out_r_in = VX_FIG8_R1 + rt + 1.0 / gw;
}

/* The full-volume DC gain of the network, which the output is scaled by so
 * that an unclipped signal at full volume keeps the chip's level. */
static double full_volume_gain(void)
{
    double shunt = 1.0 / (1.0 / VX_FIG8_POT + 1.0 / VX_FIG8_R_WIPER +
                          1.0 / VX_LM386_R_IN);
    return shunt / (VX_FIG8_R1 + shunt);
}

static void build_output_stage(vx_chip *chip)
{
    double rate = chip->core.sclock;
    build_network(chip, rate);
    pole_build(&chip->out_hp_in,
               1.0 / (2.0 * M_PI * VX_FIG8_C_IN * chip->out_r_in), rate, 1);
    pole_build(&chip->out_hp_spk,
               1.0 / (2.0 * M_PI * VX_FIG8_C_OUT * VX_FIG8_SPEAKER), rate, 1);
}

static void clear_output_stage(vx_chip *chip)
{
    pole_clear(&chip->out_hp_in);
    biquad_clear(&chip->out_net);
    pole_clear(&chip->out_hp_spk);
}

/* One sample through the whole of Figure 8, in chip units in and out. */
static double run_output_stage(vx_chip *chip, double s)
{
    const double to_volts = VX_FIG8_VOLTS_PER_UNIT;
    double v = pole_run(&chip->out_hp_in, s * to_volts);   /* at AO */
    v = biquad_run(&chip->out_net, v) * VX_LM386_GAIN;     /* LM386 out */
    v = clampd(v, -VX_LM386_SWING, VX_LM386_SWING);
    v = pole_run(&chip->out_hp_spk, v);                    /* at the speaker */
    return v / (to_volts * VX_LM386_GAIN * full_volume_gain());
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
    chip->volume = 1.0;
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
    return vx_clock_from_rc(VX_KNOB_FIXED_OHMS +
                            VX_KNOB_POT_OHMS * audio_taper(position),
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

void vx_set_output_volume(vx_chip *chip, double position)
{
    if (!chip) return;
    chip->volume = clampd(position, 0.0, 1.0);
    build_output_stage(chip);
}

double vx_output_volume(vx_chip *chip)
{
    return chip ? chip->volume : 1.0;
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
            s = run_output_stage(chip, s);
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
