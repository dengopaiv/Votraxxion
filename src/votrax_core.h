/* Votrax SC-01 / SC-01-A chip-level emulation.
 *
 * Reproduces the analog signal path: a 9-level glottal source and a noise
 * LFSR driven through seven switched-capacitor filters whose coefficients are
 * rebuilt from the chip's own interpolating parameter registers.  Tracks
 * MAME's votrax.cpp, which is Galibert's analysis of the die.
 *
 * The struct is public so a caller can embed it rather than allocate it, but
 * its fields are the chip's internal state and are not part of any contract:
 * read them for diagnostics, do not write them.
 */
#ifndef VOTRAX_CORE_H
#define VOTRAX_CORE_H

#include "votrax_filters.h"
#include "votrax_rom.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* --- configuration, set at init ------------------------------------- */
    double master_clock;
    double sclock;              /* master / 18: the analog sample rate      */
    double cclock;              /* master / 36: the chip update rate        */
    double fx_fudge;            /* final lowpass cutoff scale               */
    double closure_strength;    /* how deeply plosive closures attenuate    */
    double articulation_rate;   /* formant interpolator speed               */
    double voice_closure_ratio; /* how much closure reaches the voiced path */
    vx_mask_revision mask;
    const vx_phoneme *rom_table;

    /* --- state ----------------------------------------------------------- */
    int phone;
    int inflection;
    /* A copy, not a pointer into rom_table: phone_commit_override writes
     * arbitrary parameters here, and a pointer would make that a write to the
     * ROM. */
    vx_phoneme rom;

    /* Interpolation registers, 8-bit */
    int cur_fa, cur_fc, cur_va;
    int cur_f1, cur_f2, cur_f2q, cur_f3;

    /* Values committed to the filters, 4-bit */
    int filt_fa, filt_fc, filt_va;
    int filt_f1, filt_f2, filt_f2q, filt_f3;

    /* Timing */
    int phonetick, ticks, pitch, closure;
    int cur_closure;
    int update_counter;
    int sample_count;

    /* Noise LFSR */
    int noise;
    int cur_noise;

    /* Filter coefficients */
    double f1_a[4], f1_b[4];
    double f2v_a[4], f2v_b[4];
    double f3_a[4], f3_b[4];
    double f4_a[4], f4_b[4];
    double ns_a[3], ns_b[3];
    double f2n_a[2], f2n_b[2];
    double fx_a[2], fx_b[2];

    /* Filter histories */
    double f1_xh[4], f1_yh[3];
    double f2v_xh[4], f2v_yh[3];
    double f3_xh[4], f3_yh[3];
    double f4_xh[4], f4_yh[3];
    double ns_xh[3], ns_yh[2];
    double f2n_xh[2], f2n_yh[1];
    double fx_xh[2], fx_yh[1];
} vx_core;

/* Initialise a core and reset it.
 *
 * `master_clock` is the external clock in Hz; nominal 720 000, and the
 * datasheet endorses varying it (smaller is slower and lower, larger faster
 * and higher).
 *
 * `fx_fudge` scales the final-stage lowpass cutoff.  150/4000 matches MAME and
 * recordings of real chips; 1.0 is the as-schematic 150 Hz.
 *
 * `closure_strength` scales how deeply plosive closures attenuate the output.
 * 1.0 reproduces MAME's curve, 0.0 disables the dip so plosives lose their
 * punch, above 1.0 exaggerates it.
 *
 * `articulation_rate` scales the formant interpolator.  1.0 is SC-01 native
 * (an implicit 1/8 decay per tick); higher moves faster between targets, which
 * is what the SSI-263's articulation register did and the SC-01 could not.
 *
 * `voice_closure_ratio` (0..1) is how much of the closure dip reaches the
 * voiced path.  1.0 is SC-01 native, where closure mutes the mixed output --
 * right for voiceless stops, but it silences the voiced stops /b/ /d/ /g/,
 * which should keep their buzz.  0.0 lets voicing continue through a closure.
 * The noise path always takes the full dip.
 */
void vx_core_init(vx_core *c, double master_clock, double fx_fudge,
                  double closure_strength, double articulation_rate,
                  double voice_closure_ratio, vx_mask_revision mask);

/* Instant silence: clears interpolation state and re-latches STOP. */
void vx_core_reset(vx_core *c);

/* Switch mask ROM.  Re-points the table; the phone being voiced keeps the
 * parameters it was committed with, so this takes effect from the next
 * commit. */
void vx_core_set_mask(vx_core *c, vx_mask_revision mask);

/* Latch a phone and start voicing it. */
void vx_core_phone_commit(vx_core *c, int phone, int inflection);

/* The same, with the ROM lookup bypassed: all twelve parameters come from the
 * caller, which turns the chip into a formant instrument. */
void vx_core_phone_commit_override(vx_core *c, int phone, int inflection,
                                   const vx_phoneme *params);

/* One output sample, in roughly -1..1. */
double vx_core_generate_one_sample(vx_core *c);

/* Nonzero once the current phone has run its 16 ticks. */
int vx_core_phone_done(const vx_core *c);

/* How long a phone runs, in samples, if left alone.
 *
 * Exact and closed-form, not measured: a phone lasts 16 ticks, a tick is
 * (4 * duration + 1) chip updates, and a chip update is two samples.  Verified
 * against all 64 phones on both masks.  `duration` lives in word0, which is
 * identical between the revisions, so this does not vary with the mask.
 *
 * This is what makes constant-pitch rate control possible with no measurement
 * pass at startup: to speak faster, hold each phone for phone_samples()/speed
 * and commit the next early.  The glottal oscillator runs off the master clock
 * and never sees any of it, so tempo moves and pitch does not. */
int vx_core_phone_samples(const vx_core *c, int phone);

#ifdef __cplusplus
}
#endif

#endif /* VOTRAX_CORE_H */
