/* The chip -- see votrax_core.h. */

#include "votrax_core.h"

#include <string.h>

/* The 9-level stepped glottal waveform, from MAME's die analysis. */
static const double GLOTTAL[9] = {
    0.0, -4.0 / 7.0, 1.0, 6.0 / 7.0, 5.0 / 7.0,
    4.0 / 7.0, 3.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0
};

/* --- filter construction -------------------------------------------------- */

static void build_fixed_filters(vx_core *c)
{
    vx_build_standard_filter(c->f4_a, c->f4_b, c->sclock, c->cclock,
                             0, 28810, 1165, 21457, 8558, 7289);
    vx_build_lowpass_filter(c->fx_a, c->fx_b, c->sclock, c->cclock,
                            1122, 23131, c->fx_fudge);
    vx_build_noise_shaper_filter(c->ns_a, c->ns_b, c->sclock, c->cclock,
                                 15500, 14854, 8450, 9523, 14083);
}

/* The three formant sections and the noise injection path, rebuilt whenever
 * the parameter registers commit.  The capacitor arrays are the die's own
 * binary-weighted banks: a control bit switches its capacitor in. */
static void build_variable_filters(vx_core *c)
{
    static const double f1_caps[4]  = {2546, 4973, 9861, 19724};
    static const double f2q_caps[4] = {1390, 2965, 5875, 11297};
    static const double f2_caps[5]  = {833, 1663, 3164, 6327, 12654};
    static const double f3_caps[4]  = {2226, 4485, 9056, 18111};

    double f1_c3 = 2280 + vx_bits_to_caps(c->filt_f1, f1_caps, 4);
    double f2v_c2t = 829 + vx_bits_to_caps(c->filt_f2q, f2q_caps, 4);
    double f2v_c3 = 2352 + vx_bits_to_caps(c->filt_f2, f2_caps, 5);
    double f3_c3 = 8480 + vx_bits_to_caps(c->filt_f3, f3_caps, 4);

    vx_build_standard_filter(c->f1_a, c->f1_b, c->sclock, c->cclock,
                             11247, 11797, 949, 52067, f1_c3, 166272);
    vx_build_standard_filter(c->f2v_a, c->f2v_b, c->sclock, c->cclock,
                             24840, 29154, f2v_c2t, 38180, f2v_c3, 34270);
    vx_build_standard_filter(c->f3_a, c->f3_b, c->sclock, c->cclock,
                             0, 17594, 868, 18828, f3_c3, 50019);
    vx_build_injection_filter(c->f2n_a, c->f2n_b, c->sclock, c->cclock,
                              29154, f2v_c2t, 38180, f2v_c3, 34270);
}

/* --- lifecycle ------------------------------------------------------------ */

void vx_core_reset(vx_core *c)
{
    c->phone = 0x3F;   /* STOP */
    c->inflection = 0;
    c->rom = c->rom_table[0x3F];

    c->cur_fa = c->cur_fc = c->cur_va = 0;
    c->cur_f1 = c->cur_f2 = c->cur_f2q = c->cur_f3 = 0;

    c->filt_fa = c->filt_fc = c->filt_va = 0;
    c->filt_f1 = c->filt_f2 = c->filt_f2q = c->filt_f3 = 0;

    c->phonetick = 0;
    c->ticks = 0;
    c->pitch = 0;
    c->closure = 0;
    c->cur_closure = 1;
    c->update_counter = 0;
    c->sample_count = 0;

    c->noise = 0;
    c->cur_noise = 0;

    memset(c->f1_xh, 0, sizeof c->f1_xh);
    memset(c->f1_yh, 0, sizeof c->f1_yh);
    memset(c->f2v_xh, 0, sizeof c->f2v_xh);
    memset(c->f2v_yh, 0, sizeof c->f2v_yh);
    memset(c->f3_xh, 0, sizeof c->f3_xh);
    memset(c->f3_yh, 0, sizeof c->f3_yh);
    memset(c->f4_xh, 0, sizeof c->f4_xh);
    memset(c->f4_yh, 0, sizeof c->f4_yh);
    memset(c->ns_xh, 0, sizeof c->ns_xh);
    memset(c->ns_yh, 0, sizeof c->ns_yh);
    memset(c->f2n_xh, 0, sizeof c->f2n_xh);
    memset(c->f2n_yh, 0, sizeof c->f2n_yh);
    memset(c->fx_xh, 0, sizeof c->fx_xh);
    memset(c->fx_yh, 0, sizeof c->fx_yh);

    build_fixed_filters(c);
    build_variable_filters(c);
}

void vx_core_init(vx_core *c, double master_clock, double fx_fudge,
                  double closure_strength, double articulation_rate,
                  double voice_closure_ratio, vx_mask_revision mask)
{
    c->master_clock = master_clock;
    c->sclock = vx_sclock_from_master(master_clock);
    c->cclock = vx_cclock_from_master(master_clock);
    c->fx_fudge = fx_fudge;
    c->closure_strength = closure_strength;
    c->articulation_rate = articulation_rate;
    c->voice_closure_ratio = voice_closure_ratio;
    c->mask = mask;
    c->rom_table = vx_rom_table(mask);
    vx_core_reset(c);
}

void vx_core_set_mask(vx_core *c, vx_mask_revision mask)
{
    c->mask = mask;
    c->rom_table = vx_rom_table(mask);
}

void vx_core_phone_commit(vx_core *c, int phone, int inflection)
{
    c->phone = phone & 0x3F;
    c->inflection = inflection & 0x03;
    c->rom = c->rom_table[c->phone];
    c->phonetick = 0;
    c->ticks = 0;
    if (c->rom.cld == 0)
        c->cur_closure = c->rom.closure;
}

void vx_core_phone_commit_override(vx_core *c, int phone, int inflection,
                                   const vx_phoneme *params)
{
    c->phone = phone & 0x3F;
    c->inflection = inflection & 0x03;
    c->rom = *params;
    c->phonetick = 0;
    c->ticks = 0;
    if (c->rom.cld == 0)
        c->cur_closure = c->rom.closure;
}

int vx_core_phone_done(const vx_core *c)
{
    return c->ticks >= 0x10;
}

int vx_core_phone_samples(const vx_core *c, int phone)
{
    return 32 * (4 * c->rom_table[phone & 0x3F].duration + 1);
}

/* --- the digital half ----------------------------------------------------- */

/* One step of a parameter register toward its target.
 *
 * The native path is the chip's own fixed-point interpolator: the register
 * decays by 1/8 and takes twice the target each tick, so it settles at 16x the
 * target in 8-bit space.  The scaled path generalises that to
 * next = reg + alpha*(16*target - reg) with alpha = articulation_rate/8, which
 * is identical at rate 1.0 but is not the chip -- the SC-01 has no such
 * control. */
static void interpolate(const vx_core *c, int *reg, int target)
{
    if (c->articulation_rate == 1.0) {
        *reg = (*reg - (*reg >> 3) + (target << 1)) & 0xFF;
    } else {
        double alpha = 0.125 * c->articulation_rate;
        double steady, next;
        if (alpha > 1.0) alpha = 1.0;
        if (alpha < 0.0) alpha = 0.0;
        steady = 16.0 * (double)target;
        next = (double)*reg + alpha * (steady - (double)*reg);
        if (next < 0.0) next = 0.0;
        if (next > 255.0) next = 255.0;
        *reg = (int)next & 0xFF;
    }
}

static void interpolate_formants(vx_core *c)
{
    interpolate(c, &c->cur_fc, c->rom.fc);
    interpolate(c, &c->cur_f1, c->rom.f1);
    interpolate(c, &c->cur_f2, c->rom.f2);
    interpolate(c, &c->cur_f2q, c->rom.f2q);
    interpolate(c, &c->cur_f3, c->rom.f3);
}

static void commit_filters(vx_core *c)
{
    c->filt_f1  = c->cur_f1 >> 4;
    c->filt_va  = c->cur_va >> 4;
    c->filt_f2  = c->cur_f2 >> 3;
    c->filt_fc  = c->cur_fc >> 4;
    c->filt_f2q = c->cur_f2q >> 4;
    c->filt_f3  = c->cur_f3 >> 4;
    c->filt_fa  = c->cur_fa >> 4;
    build_variable_filters(c);
}

static void chip_update(vx_core *c)
{
    int pitch_target;
    int tick_625, tick_208;
    int inp;

    /* Duration counter */
    if (c->ticks != 0x10) {
        c->phonetick++;
        if (c->phonetick == ((c->rom.duration << 2) | 1)) {
            c->phonetick = 0;
            c->ticks++;
            if (c->ticks == c->rom.cld)
                c->cur_closure = c->rom.closure;
        }
    }

    c->update_counter = (c->update_counter + 1) % 0x30;
    tick_625 = !(c->update_counter & 0xF);
    tick_208 = (c->update_counter == 0x28);

    if (tick_208 && (!c->rom.pause || !(c->filt_fa || c->filt_va)))
        interpolate_formants(c);

    if (tick_625) {
        if (c->ticks >= c->rom.vd)
            interpolate(c, &c->cur_fa, c->rom.fa);
        if (c->ticks >= c->rom.cld)
            interpolate(c, &c->cur_va, c->rom.va);
    }

    if (!c->cur_closure && (c->filt_fa || c->filt_va))
        c->closure = 0;
    else if (c->closure != (7 << 2))
        c->closure++;

    /* Pitch counter */
    c->pitch = (c->pitch + 1) & 0xFF;
    pitch_target = ((0xE0 ^ (c->inflection << 5) ^ (c->filt_f1 << 1)) + 2);
    if (c->pitch == pitch_target)
        c->pitch = 0;

    if ((c->pitch & 0xF9) == 0x08)
        commit_filters(c);

    /* Noise LFSR.
     *
     * The `1 ||` is MAME's, and it reads like a condition somebody disabled
     * and forgot to remove.  It is kept verbatim, short-circuit and all: the
     * dead half changes nothing today, but deleting it would leave a later
     * reader thinking the condition was always filt_fa and "restoring" a bug
     * that is not there. */
    inp = (1 || c->filt_fa) && c->cur_noise && (c->noise != 0x7FFF);
    c->noise = ((c->noise << 1) & 0x7FFE) | (inp ? 1 : 0);
    c->cur_noise = !(((c->noise >> 14) ^ (c->noise >> 13)) & 1);
}

/* --- the analog half ------------------------------------------------------ */

static double analog_calc(vx_core *c)
{
    int glot_idx = c->pitch >> 3;
    double glottal = (glot_idx < 9) ? GLOTTAL[glot_idx] : 0.0;

    /* Closure attenuation.  At voice_closure_ratio 1.0 this is the SC-01's own
     * path, where closure is applied once at the output.  Below 1.0 it is
     * applied separately to the voice and noise inputs so the voiced path can
     * survive the dip -- a deliberate divergence from the hardware, to get the
     * SSI-263's audible voiced stops. */
    double mame_atten = (7 ^ (c->closure >> 2)) / 7.0;
    double noise_closure_atten =
        1.0 - c->closure_strength * (1.0 - mame_atten);
    double voice_closure_atten =
        1.0 - c->voice_closure_ratio * c->closure_strength * (1.0 - mame_atten);
    int selective = (c->voice_closure_ratio < 1.0);

    double voice, f1_out, f2v_out;
    double noise_raw, noise_in, ns_out, noise_f2n_in, f2n_out, noise_direct;
    double combined, f3_out, f3_plus_noise, f4_out, closure_out, fx_out;
    int noise_gate;

    /* Voice path: glottal * va/15 -> F1 -> F2v */
    voice = glottal * (c->filt_va / 15.0);
    if (selective) voice *= voice_closure_atten;

    vx_shift_hist(voice, c->f1_xh, 4);
    f1_out = vx_apply_filter(c->f1_xh, c->f1_yh, c->f1_a, c->f1_b, 4, 4);
    vx_shift_hist(f1_out, c->f1_yh, 3);

    vx_shift_hist(f1_out, c->f2v_xh, 4);
    f2v_out = vx_apply_filter(c->f2v_xh, c->f2v_yh, c->f2v_a, c->f2v_b, 4, 4);
    vx_shift_hist(f2v_out, c->f2v_yh, 3);

    /* Noise path */
    noise_gate = (c->pitch & 0x40) ? c->cur_noise : 0;
    noise_raw = 1e4 * (noise_gate ? 1.0 : -1.0);
    noise_in = noise_raw * (c->filt_fa / 15.0);
    if (selective) noise_in *= noise_closure_atten;

    vx_shift_hist(noise_in, c->ns_xh, 3);
    ns_out = vx_apply_filter(c->ns_xh, c->ns_yh, c->ns_a, c->ns_b, 3, 3);
    vx_shift_hist(ns_out, c->ns_yh, 2);

    noise_f2n_in = ns_out * (c->filt_fc / 15.0);
    vx_shift_hist(noise_f2n_in, c->f2n_xh, 2);
    f2n_out = vx_apply_filter(c->f2n_xh, c->f2n_yh, c->f2n_a, c->f2n_b, 2, 2);
    vx_shift_hist(f2n_out, c->f2n_yh, 1);

    noise_direct = ns_out * (5.0 + (15 ^ c->filt_fc)) / 20.0;

    combined = f2v_out + f2n_out;

    vx_shift_hist(combined, c->f3_xh, 4);
    f3_out = vx_apply_filter(c->f3_xh, c->f3_yh, c->f3_a, c->f3_b, 4, 4);
    vx_shift_hist(f3_out, c->f3_yh, 3);

    f3_plus_noise = f3_out + noise_direct;

    vx_shift_hist(f3_plus_noise, c->f4_xh, 4);
    f4_out = vx_apply_filter(c->f4_xh, c->f4_yh, c->f4_a, c->f4_b, 4, 4);
    vx_shift_hist(f4_out, c->f4_yh, 3);

    /* Native mode applies closure once, here at the output, which keeps this
     * byte-for-byte with the original path.  Selective mode already applied it
     * at the inputs. */
    closure_out = selective ? f4_out : (f4_out * noise_closure_atten);

    vx_shift_hist(closure_out, c->fx_xh, 2);
    fx_out = vx_apply_filter(c->fx_xh, c->fx_yh, c->fx_a, c->fx_b, 2, 2);
    vx_shift_hist(fx_out, c->fx_yh, 1);

    return fx_out * 0.35;
}

double vx_core_generate_one_sample(vx_core *c)
{
    c->sample_count++;
    if (c->sample_count % 2 == 0)
        chip_update(c);
    return analog_calc(c);
}
