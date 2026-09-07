/* Filter construction -- see votrax_filters.h. */

#include "votrax_filters.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double vx_sclock_from_master(double master_clock)
{
    return master_clock / VX_SCLOCK_DIVIDER;
}

double vx_cclock_from_master(double master_clock)
{
    return master_clock / VX_CCLOCK_DIVIDER;
}

double vx_bits_to_caps(int value, const double *caps, int ncaps)
{
    double total = 0.0;
    int i;
    for (i = 0; i < ncaps; i++) {
        if (value & (1 << i))
            total += caps[i];
    }
    return total;
}

void vx_build_standard_filter(double *a, double *b,
                              double sclock, double cclock,
                              double c1t, double c1b,
                              double c2t, double c2b,
                              double c3, double c4)
{
    double k0 = c1t / (cclock * c1b);
    double k1 = c4 * c2t / (cclock * c1b * c3);
    double k2 = c4 * c2b / (cclock * cclock * c1b * c3);

    double fpeak = sqrt(fabs(k0 * k1 - k2)) / (2.0 * M_PI * k2);
    double zc = 2.0 * M_PI * fpeak / tan(M_PI * fpeak / sclock);

    double m0 = zc * k0;
    double m1 = zc * k1;
    double m2 = zc * zc * k2;

    double inv;
    int i;

    a[0] = 1.0 + m0;
    a[1] = 3.0 + m0;
    a[2] = 3.0 - m0;
    a[3] = 1.0 - m0;
    b[0] = 1.0 + m1 + m2;
    b[1] = 3.0 + m1 - m2;
    b[2] = 3.0 - m1 - m2;
    b[3] = 1.0 - m1 + m2;

    inv = 1.0 / b[0];
    for (i = 0; i < 4; i++) {
        a[i] *= inv;
        b[i] *= inv;
    }
}

void vx_build_noise_shaper_filter(double *a, double *b,
                                  double sclock, double cclock,
                                  double c1, double c2t,
                                  double c2b, double c3, double c4)
{
    /* Note k1: cclock is in the numerator here and in a denominator in every
     * other builder.  That is how MAME has it, and it is the whole reason the
     * fricative path is the one thing a clock change actually alters. */
    double k0 = c2t * c3 * c2b / c4;
    double k1 = c2t * (cclock * c2b);
    double k2 = c1 * c2t * c3 / (cclock * c4);

    double fpeak = sqrt(1.0 / k2) / (2.0 * M_PI);
    double zc = 2.0 * M_PI * fpeak / tan(M_PI * fpeak / sclock);

    double m0 = zc * k0;
    double m1 = zc * k1;
    double m2 = zc * zc * k2;

    double inv;
    int i;

    a[0] = m0;
    a[1] = 0.0;
    a[2] = -m0;
    b[0] = 1.0 + m1 + m2;
    b[1] = 2.0 - 2.0 * m2;
    b[2] = 1.0 - m1 + m2;

    inv = 1.0 / b[0];
    for (i = 0; i < 3; i++) {
        a[i] *= inv;
        b[i] *= inv;
    }
}

void vx_build_lowpass_filter(double *a, double *b,
                             double sclock, double cclock,
                             double c1t, double c1b, double fx_fudge)
{
    double k = c1b / (cclock * c1t) * fx_fudge;
    double fpeak = 1.0 / (2.0 * M_PI * k);
    double zc = 2.0 * M_PI * fpeak / tan(M_PI * fpeak / sclock);
    double m = zc * k;

    a[0] = 1.0 / (1.0 + m);
    a[1] = 0.0;
    b[0] = 1.0;
    b[1] = (1.0 - m) / (1.0 + m);
}

void vx_build_injection_filter(double *a, double *b,
                               double sclock, double cclock,
                               double c1b, double c2t,
                               double c2b, double c3, double c4)
{
    double k0 = c2t / (cclock * c1b);
    double k1 = c4 * c2t / (cclock * c1b * c3);
    double k2 = c4 * c2b / (cclock * cclock * c1b * c3);
    double c, denom;

    if (k1 <= 0.0) {
        a[0] = 0.0;
        a[1] = 0.0;
        b[0] = 1.0;
        b[1] = 0.0;
        return;
    }

    c = 2.0 * sclock;   /* the bilinear transform constant */
    denom = k1 + k2 * c;

    a[0] = (k0 + k2 * c) / denom;
    a[1] = (k0 - k2 * c) / denom;
    b[0] = 1.0;
    b[1] = (k1 - k2 * c) / denom;
}

double vx_apply_filter(const double *x_hist, const double *y_hist,
                       const double *a, const double *b, int na, int nb)
{
    double result = 0.0;
    int i;
    for (i = 0; i < na; i++)
        result += a[i] * x_hist[i];
    for (i = 1; i < nb; i++)
        result -= b[i] * y_hist[i - 1];
    return result;
}

void vx_shift_hist(double val, double *hist, int n)
{
    int i;
    for (i = n - 1; i > 0; i--)
        hist[i] = hist[i - 1];
    hist[0] = val;
}
