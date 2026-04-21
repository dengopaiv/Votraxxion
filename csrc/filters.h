// Votrax SC-01A analog filter construction using bilinear z-transform.
// All filter designs and capacitor values from MAME votrax.cpp die analysis.
//
// Master clock is now a runtime parameter (see SC-01 datasheet 1980, which
// explicitly endorses varying it as a sound-design technique: Fig 6 pot
// control, Fig 7 DAC injection). Clock relationships are fixed by the chip:
//   SCLOCK = master_clock / 18   (analog sample rate)
//   CCLOCK = master_clock / 36   (chip update rate)
#pragma once

#include <cmath>
#include <cstring>
#include <array>

static constexpr double DEFAULT_MASTER_CLOCK = 720000.0;
static constexpr double SCLOCK_DIVIDER = 18.0;
static constexpr double CCLOCK_DIVIDER = 36.0;

inline double sclock_from_master(double master_clock) {
    return master_clock / SCLOCK_DIVIDER;
}

inline double cclock_from_master(double master_clock) {
    return master_clock / CCLOCK_DIVIDER;
}

// Convert a multi-bit value to a summed capacitance.
// Each bit in value selects the corresponding capacitor from caps.
inline double bits_to_caps(int value, const double* caps, int ncaps) {
    double total = 0.0;
    for (int i = 0; i < ncaps; i++) {
        if (value & (1 << i))
            total += caps[i];
    }
    return total;
}

// Build a 3rd-order formant filter (F1, F2v, F3, F4).
// H(s) = (1 + k0*s) / (1 + k1*s + k2*s^2)
// Returns coefficients in a[4] and b[4], normalized so b[0]=1.
inline void build_standard_filter(double* a, double* b,
                                   double sclock, double cclock,
                                   double c1t, double c1b,
                                   double c2t, double c2b,
                                   double c3, double c4) {
    double k0 = c1t / (cclock * c1b);
    double k1 = c4 * c2t / (cclock * c1b * c3);
    double k2 = c4 * c2b / (cclock * cclock * c1b * c3);

    double fpeak = std::sqrt(std::fabs(k0 * k1 - k2)) / (2.0 * M_PI * k2);
    double zc = 2.0 * M_PI * fpeak / std::tan(M_PI * fpeak / sclock);

    double m0 = zc * k0;
    double m1 = zc * k1;
    double m2 = zc * zc * k2;

    a[0] = 1.0 + m0;
    a[1] = 3.0 + m0;
    a[2] = 3.0 - m0;
    a[3] = 1.0 - m0;
    b[0] = 1.0 + m1 + m2;
    b[1] = 3.0 + m1 - m2;
    b[2] = 3.0 - m1 - m2;
    b[3] = 1.0 - m1 + m2;

    // Normalize so b[0] = 1
    double inv = 1.0 / b[0];
    for (int i = 0; i < 4; i++) { a[i] *= inv; b[i] *= inv; }
}

// Build the noise shaper filter (2nd-order bandpass).
// H(s) = k0*s / (1 + k1*s + k2*s^2)
// Returns coefficients in a[3] and b[3], normalized so b[0]=1.
inline void build_noise_shaper_filter(double* a, double* b,
                                       double sclock, double cclock,
                                       double c1, double c2t,
                                       double c2b, double c3, double c4) {
    double k0 = c2t * c3 * c2b / c4;
    double k1 = c2t * (cclock * c2b);
    double k2 = c1 * c2t * c3 / (cclock * c4);

    double fpeak = std::sqrt(1.0 / k2) / (2.0 * M_PI);
    double zc = 2.0 * M_PI * fpeak / std::tan(M_PI * fpeak / sclock);

    double m0 = zc * k0;
    double m1 = zc * k1;
    double m2 = zc * zc * k2;

    a[0] = m0;
    a[1] = 0.0;
    a[2] = -m0;
    b[0] = 1.0 + m1 + m2;
    b[1] = 2.0 - 2.0 * m2;
    b[2] = 1.0 - m1 + m2;

    double inv = 1.0 / b[0];
    for (int i = 0; i < 3; i++) { a[i] *= inv; b[i] *= inv; }
}

// Build the final output lowpass filter (FX).
// Per MAME: the on-die capacitor values give a cutoff ~150 Hz, but actual chip
// recordings are ~4 kHz — so we fuzz by that ratio. 4 kHz is the authentic
// bandwidth (see Tech overview.md Part 3). `fx_fudge` defaults to 150/4000
// (matches MAME); passing 1.0 would restore the "as-schematic" 150 Hz behavior.
inline void build_lowpass_filter(double* a, double* b,
                                  double sclock, double cclock,
                                  double c1t, double c1b,
                                  double fx_fudge = 150.0 / 4000.0) {
    double k = c1b / (cclock * c1t) * fx_fudge;
    double fpeak = 1.0 / (2.0 * M_PI * k);
    double zc = 2.0 * M_PI * fpeak / std::tan(M_PI * fpeak / sclock);
    double m = zc * k;

    a[0] = 1.0 / (1.0 + m);
    a[1] = 0.0;
    b[0] = 1.0;
    b[1] = (1.0 - m) / (1.0 + m);
}

// Build the noise injection filter (F2n) via pole-reflected bilinear transform.
// The Python implementation reflects the RHP pole to get a stable filter,
// unlike MAME which neutralizes it.
inline void build_injection_filter(double* a, double* b,
                                    double sclock, double cclock,
                                    double c1b, double c2t,
                                    double c2b, double c3, double c4) {
    double k0 = c2t / (cclock * c1b);
    double k1 = c4 * c2t / (cclock * c1b * c3);
    double k2 = c4 * c2b / (cclock * cclock * c1b * c3);

    if (k1 <= 0.0) {
        a[0] = 0.0; a[1] = 0.0;
        b[0] = 1.0; b[1] = 0.0;
        return;
    }

    double c = 2.0 * sclock;  // bilinear transform constant
    double denom = k1 + k2 * c;

    a[0] = (k0 + k2 * c) / denom;
    a[1] = (k0 - k2 * c) / denom;
    b[0] = 1.0;
    b[1] = (k1 - k2 * c) / denom;
}

// Apply IIR filter: compute one output sample.
// Template version for compile-time known sizes.
template<int NA, int NB>
inline double apply_filter(const double* x_hist, const double* y_hist,
                            const double* a, const double* b) {
    double result = 0.0;
    for (int i = 0; i < NA; i++)
        result += a[i] * x_hist[i];
    for (int i = 1; i < NB; i++)
        result -= b[i] * y_hist[i - 1];
    return result;
}

// Push a new value into a history buffer, shifting old values right.
// Template version for compile-time known sizes.
template<int N>
inline void shift_hist(double val, double* hist) {
    for (int i = N - 1; i > 0; i--)
        hist[i] = hist[i - 1];
    hist[0] = val;
}
