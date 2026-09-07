/* Votrax SC-01 analog filter construction, by bilinear z-transform.
 *
 * The chip's analog half is seven switched-capacitor filters.  Their component
 * values are on the die, so the builders below take capacitances -- literally
 * the capacitor sizes Galibert measured off the photographs -- and turn them
 * into digital coefficients for whatever sample rate the master clock implies.
 * All designs and values track MAME's votrax.cpp die analysis.
 *
 * The master clock is a runtime parameter.  The 1980 datasheet explicitly
 * endorses varying it as a sound-design technique (Fig 6 pot control, Fig 7
 * DAC injection).  The two derived clocks are fixed ratios of it:
 *
 *   sclock = master / 18    the analog sample rate
 *   cclock = master / 36    the chip update rate
 *
 * A consequence worth knowing: because those scale together, most of these
 * builders produce coefficients that do not depend on the master clock at all
 * -- moving the clock moves the playback rate, not the waveform.  The one
 * exception is the noise shaper, whose k1 has cclock in the numerator where
 * every other term has it in a denominator, so the fricative path does change.
 * That asymmetry is MAME's and is reproduced deliberately.
 */
#ifndef VOTRAX_FILTERS_H
#define VOTRAX_FILTERS_H

#ifdef __cplusplus
extern "C" {
#endif

#define VX_DEFAULT_MASTER_CLOCK 720000.0
#define VX_SCLOCK_DIVIDER 18.0
#define VX_CCLOCK_DIVIDER 36.0

double vx_sclock_from_master(double master_clock);
double vx_cclock_from_master(double master_clock);

/* Sum the capacitors a multi-bit control value selects. */
double vx_bits_to_caps(int value, const double *caps, int ncaps);

/* A 3rd-order formant section (F1, F2v, F3, F4).
 *   H(s) = (1 + k0 s) / (1 + k1 s + k2 s^2)
 * Writes a[4] and b[4], normalized so b[0] = 1. */
void vx_build_standard_filter(double *a, double *b,
                              double sclock, double cclock,
                              double c1t, double c1b,
                              double c2t, double c2b,
                              double c3, double c4);

/* The noise shaper: a 2nd-order bandpass.
 *   H(s) = k0 s / (1 + k1 s + k2 s^2)
 * Writes a[3] and b[3], normalized so b[0] = 1. */
void vx_build_noise_shaper_filter(double *a, double *b,
                                  double sclock, double cclock,
                                  double c1, double c2t,
                                  double c2b, double c3, double c4);

/* The final output lowpass (FX).  The on-die capacitors give a cutoff around
 * 150 Hz, but recordings of real chips are around 4 kHz, so the cutoff is
 * scaled by `fx_fudge` -- 150/4000 reproduces MAME and the recordings, 1.0
 * restores the as-schematic behaviour.  Writes a[2] and b[2]. */
void vx_build_lowpass_filter(double *a, double *b,
                             double sclock, double cclock,
                             double c1t, double c1b, double fx_fudge);

/* The noise injection path (F2n), by pole-reflected bilinear transform: the
 * right-half-plane pole is reflected to get a stable filter, where MAME
 * neutralizes it instead.  Writes a[2] and b[2]. */
void vx_build_injection_filter(double *a, double *b,
                               double sclock, double cclock,
                               double c1b, double c2t,
                               double c2b, double c3, double c4);

/* One output sample of an IIR section from its history buffers.
 *
 * These were templates on the sizes in the C++.  They are ordinary functions
 * here: every call site passes a constant, so the compiler unrolls them just
 * the same, and the generated code is identical. */
double vx_apply_filter(const double *x_hist, const double *y_hist,
                       const double *a, const double *b, int na, int nb);

/* Push a value into a history buffer, shifting the old ones along. */
void vx_shift_hist(double val, double *hist, int n);

#ifdef __cplusplus
}
#endif

#endif /* VOTRAX_FILTERS_H */
