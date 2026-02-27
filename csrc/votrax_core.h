// Votrax SC-01A chip-level emulation — C++ core.
// Faithfully reproduces the MAME votrax.cpp analog signal path.
// Optional enhanced mode: KLGLOTT88 glottal source + PolyBLEP + jitter/shimmer.
#pragma once

#include "filters.h"
#include "rom_data.h"
#include <random>

// 9-level glottal waveform from MAME
static constexpr double GLOTTAL[9] = {
    0.0, -4.0/7.0, 1.0, 6.0/7.0, 5.0/7.0, 4.0/7.0, 3.0/7.0, 2.0/7.0, 1.0/7.0
};

// KLGLOTT88 polynomial glottal pulse.
// OQ = open quotient (fraction of period glottis is open), SQ = speed quotient.
// Produces smooth waveform matching SC-01A duty cycle.
inline double klglott88(double phase, double oq = 0.55, double sq = 2.0) {
    if (phase >= oq) return 0.0;  // closed phase
    double opening_end = oq / (1.0 + sq);
    if (phase < opening_end) {
        // Opening: smooth cubic rise (Hermite)
        double t = phase / opening_end;
        return 3.0 * t * t - 2.0 * t * t * t;
    } else {
        // Closing: quadratic fall
        double t = (phase - opening_end) / (oq - opening_end);
        return 1.0 - t * t;
    }
}

// PolyBLEP correction at discontinuities.
// t = phase (0 to 1), dt = 1/period_in_samples.
inline double polyblep(double t, double dt) {
    if (t < dt) {
        t /= dt;
        return t + t - t * t - 1.0;
    }
    if (t > 1.0 - dt) {
        t = (t - 1.0) / dt;
        return t * t + t + t + 1.0;
    }
    return 0.0;
}

class VotraxSC01ACore {
public:
    VotraxSC01ACore(bool enhanced = false)
        : m_enhanced(enhanced), m_rng(42) { reset(); }

    void reset() {
        m_phone = 0x3F;  // STOP
        m_inflection = 0;
        m_rom = ROM_DATA[0x3F];

        // Interpolation registers
        m_cur_fa = m_cur_fc = m_cur_va = 0;
        m_cur_f1 = m_cur_f2 = m_cur_f2q = m_cur_f3 = 0;

        // Committed filter values
        m_filt_fa = m_filt_fc = m_filt_va = 0;
        m_filt_f1 = m_filt_f2 = m_filt_f2q = m_filt_f3 = 0;

        // Timing
        m_phonetick = 0;
        m_ticks = 0;
        m_pitch = 0;
        m_closure = 0;
        m_cur_closure = true;
        m_update_counter = 0;
        m_sample_count = 0;

        // Noise LFSR
        m_noise = 0;
        m_cur_noise = false;

        // Enhanced mode state
        m_pitch_period = 128;  // default period

        // Clear all filter histories
        std::memset(m_f1_xh, 0, sizeof(m_f1_xh));
        std::memset(m_f1_yh, 0, sizeof(m_f1_yh));
        std::memset(m_f2v_xh, 0, sizeof(m_f2v_xh));
        std::memset(m_f2v_yh, 0, sizeof(m_f2v_yh));
        std::memset(m_f3_xh, 0, sizeof(m_f3_xh));
        std::memset(m_f3_yh, 0, sizeof(m_f3_yh));
        std::memset(m_f4_xh, 0, sizeof(m_f4_xh));
        std::memset(m_f4_yh, 0, sizeof(m_f4_yh));
        std::memset(m_ns_xh, 0, sizeof(m_ns_xh));
        std::memset(m_ns_yh, 0, sizeof(m_ns_yh));
        std::memset(m_f2n_xh, 0, sizeof(m_f2n_xh));
        std::memset(m_f2n_yh, 0, sizeof(m_f2n_yh));
        std::memset(m_fx_xh, 0, sizeof(m_fx_xh));
        std::memset(m_fx_yh, 0, sizeof(m_fx_yh));

        build_fixed_filters();
        build_variable_filters();
    }

    void phone_commit(int phone, int inflection = 0) {
        m_phone = phone & 0x3F;
        m_inflection = inflection & 0x03;
        m_rom = ROM_DATA[m_phone];
        m_phonetick = 0;
        m_ticks = 0;
        if (m_rom.cld == 0)
            m_cur_closure = m_rom.closure;
    }

    double generate_one_sample() {
        m_sample_count++;
        if (m_sample_count % 2 == 0)
            chip_update();
        return analog_calc();
    }

    bool phone_done() const {
        return m_ticks >= 0x10;
    }

    bool enhanced() const { return m_enhanced; }

private:
    bool m_enhanced;
    std::mt19937 m_rng;

    // --- State ---
    int m_phone;
    int m_inflection;
    PhonemeParams m_rom;

    // Interpolation registers (8-bit)
    int m_cur_fa, m_cur_fc, m_cur_va;
    int m_cur_f1, m_cur_f2, m_cur_f2q, m_cur_f3;

    // Committed filter values
    int m_filt_fa, m_filt_fc, m_filt_va;
    int m_filt_f1, m_filt_f2, m_filt_f2q, m_filt_f3;

    // Timing
    int m_phonetick, m_ticks, m_pitch, m_closure;
    bool m_cur_closure;
    int m_update_counter;
    int m_sample_count;

    // Noise LFSR
    int m_noise;
    bool m_cur_noise;

    // Enhanced mode: pitch period tracking
    int m_pitch_period;

    // Filter coefficients
    double m_f1_a[4], m_f1_b[4];
    double m_f2v_a[4], m_f2v_b[4];
    double m_f3_a[4], m_f3_b[4];
    double m_f4_a[4], m_f4_b[4];
    double m_ns_a[3], m_ns_b[3];
    double m_f2n_a[2], m_f2n_b[2];
    double m_fx_a[2], m_fx_b[2];

    // Filter histories
    double m_f1_xh[4], m_f1_yh[3];
    double m_f2v_xh[4], m_f2v_yh[3];
    double m_f3_xh[4], m_f3_yh[3];
    double m_f4_xh[4], m_f4_yh[3];
    double m_ns_xh[3], m_ns_yh[2];
    double m_f2n_xh[2], m_f2n_yh[1];
    double m_fx_xh[2], m_fx_yh[1];

    // --- Filter building ---
    void build_fixed_filters() {
        build_standard_filter(m_f4_a, m_f4_b, 0, 28810, 1165, 21457, 8558, 7289);
        build_lowpass_filter(m_fx_a, m_fx_b, 1122, 23131);
        build_noise_shaper_filter(m_ns_a, m_ns_b, 15500, 14854, 8450, 9523, 14083);
    }

    void build_variable_filters() {
        double f1_caps[] = {2546, 4973, 9861, 19724};
        double f1_c3 = 2280 + bits_to_caps(m_filt_f1, f1_caps, 4);
        build_standard_filter(m_f1_a, m_f1_b, 11247, 11797, 949, 52067, f1_c3, 166272);

        double f2q_caps[] = {1390, 2965, 5875, 11297};
        double f2_caps[] = {833, 1663, 3164, 6327, 12654};
        double f2v_c2t = 829 + bits_to_caps(m_filt_f2q, f2q_caps, 4);
        double f2v_c3 = 2352 + bits_to_caps(m_filt_f2, f2_caps, 5);
        build_standard_filter(m_f2v_a, m_f2v_b, 24840, 29154, f2v_c2t, 38180, f2v_c3, 34270);

        double f3_caps[] = {2226, 4485, 9056, 18111};
        double f3_c3 = 8480 + bits_to_caps(m_filt_f3, f3_caps, 4);
        build_standard_filter(m_f3_a, m_f3_b, 0, 17594, 868, 18828, f3_c3, 50019);

        build_injection_filter(m_f2n_a, m_f2n_b,
            29154,
            829 + bits_to_caps(m_filt_f2q, f2q_caps, 4),
            38180,
            2352 + bits_to_caps(m_filt_f2, f2_caps, 5),
            34270);
    }

    void interpolate(int& reg, int target) {
        reg = (reg - (reg >> 3) + (target << 1)) & 0xFF;
    }

    void interpolate_formants() {
        interpolate(m_cur_fc, m_rom.fc);
        interpolate(m_cur_f1, m_rom.f1);
        interpolate(m_cur_f2, m_rom.f2);
        interpolate(m_cur_f2q, m_rom.f2q);
        interpolate(m_cur_f3, m_rom.f3);
    }

    void commit_filters() {
        m_filt_f1 = m_cur_f1 >> 4;
        m_filt_va = m_cur_va >> 4;
        m_filt_f2 = m_cur_f2 >> 3;
        m_filt_fc = m_cur_fc >> 4;
        m_filt_f2q = m_cur_f2q >> 4;
        m_filt_f3 = m_cur_f3 >> 4;
        m_filt_fa = m_cur_fa >> 4;
        build_variable_filters();
    }

    void chip_update() {
        // Duration counter
        if (m_ticks != 0x10) {
            m_phonetick++;
            if (m_phonetick == ((m_rom.duration << 2) | 1)) {
                m_phonetick = 0;
                m_ticks++;
                if (m_ticks == m_rom.cld)
                    m_cur_closure = m_rom.closure;
            }
        }

        m_update_counter = (m_update_counter + 1) % 0x30;
        bool tick_625 = !(m_update_counter & 0xF);
        bool tick_208 = (m_update_counter == 0x28);

        if (tick_208 && (!m_rom.pause || !(m_filt_fa || m_filt_va)))
            interpolate_formants();

        if (tick_625) {
            if (m_ticks >= m_rom.vd)
                interpolate(m_cur_fa, m_rom.fa);
            if (m_ticks >= m_rom.cld)
                interpolate(m_cur_va, m_rom.va);
        }

        if (!m_cur_closure && (m_filt_fa || m_filt_va))
            m_closure = 0;
        else if (m_closure != (7 << 2))
            m_closure++;

        // Pitch counter
        m_pitch = (m_pitch + 1) & 0xFF;
        int pitch_target = ((0xE0 ^ (m_inflection << 5) ^ (m_filt_f1 << 1)) + 2);

        if (m_enhanced) {
            // Apply jitter: ~1.5% Gaussian perturbation
            std::normal_distribution<double> jitter_dist(0.0, 0.015 * pitch_target);
            int jittered = pitch_target + static_cast<int>(jitter_dist(m_rng));
            if (m_pitch == jittered) {
                m_pitch_period = jittered;
                m_pitch = 0;
            }
        } else {
            if (m_pitch == pitch_target) {
                m_pitch_period = pitch_target;
                m_pitch = 0;
            }
        }

        if ((m_pitch & 0xF9) == 0x08)
            commit_filters();

        // Noise LFSR
        bool inp = (true || m_filt_fa) && m_cur_noise && (m_noise != 0x7FFF);
        m_noise = ((m_noise << 1) & 0x7FFE) | (inp ? 1 : 0);
        m_cur_noise = !(((m_noise >> 14) ^ (m_noise >> 13)) & 1);
    }

    double analog_calc() {
        double glottal;

        if (m_enhanced) {
            // KLGLOTT88 polynomial pulse with PolyBLEP
            double period = (m_pitch_period > 0) ? static_cast<double>(m_pitch_period) : 128.0;
            double phase = static_cast<double>(m_pitch) / period;
            if (phase >= 1.0) phase = 0.0;

            glottal = klglott88(phase);

            // PolyBLEP at glottal closure (phase ~ OQ = 0.55)
            double dt = 1.0 / period;
            double closure_phase = phase - 0.55;
            if (closure_phase < 0.0) closure_phase += 1.0;
            glottal += polyblep(closure_phase, dt) * 0.5;

            // Shimmer: ~3% amplitude variation
            std::normal_distribution<double> shimmer_dist(1.0, 0.03);
            glottal *= shimmer_dist(m_rng);
        } else {
            // Original 9-level MAME glottal waveform
            int glot_idx = m_pitch >> 3;
            glottal = (glot_idx < 9) ? GLOTTAL[glot_idx] : 0.0;
        }

        // Voice path: glottal * va/15 -> F1 -> F2v
        double voice = glottal * (m_filt_va / 15.0);

        shift_hist<4>(voice, m_f1_xh);
        double f1_out = apply_filter<4, 4>(m_f1_xh, m_f1_yh, m_f1_a, m_f1_b);
        shift_hist<3>(f1_out, m_f1_yh);

        shift_hist<4>(f1_out, m_f2v_xh);
        double f2v_out = apply_filter<4, 4>(m_f2v_xh, m_f2v_yh, m_f2v_a, m_f2v_b);
        shift_hist<3>(f2v_out, m_f2v_yh);

        // Noise path
        bool noise_gate = (m_pitch & 0x40) ? m_cur_noise : false;
        double noise_raw = 1e4 * (noise_gate ? 1.0 : -1.0);
        double noise_in = noise_raw * (m_filt_fa / 15.0);

        shift_hist<3>(noise_in, m_ns_xh);
        double ns_out = apply_filter<3, 3>(m_ns_xh, m_ns_yh, m_ns_a, m_ns_b);
        shift_hist<2>(ns_out, m_ns_yh);

        double noise_f2n_in = ns_out * (m_filt_fc / 15.0);
        shift_hist<2>(noise_f2n_in, m_f2n_xh);
        double f2n_out = apply_filter<2, 2>(m_f2n_xh, m_f2n_yh, m_f2n_a, m_f2n_b);
        shift_hist<1>(f2n_out, m_f2n_yh);

        double noise_direct = ns_out * (5.0 + (15 ^ m_filt_fc)) / 20.0;

        double combined = f2v_out + f2n_out;

        shift_hist<4>(combined, m_f3_xh);
        double f3_out = apply_filter<4, 4>(m_f3_xh, m_f3_yh, m_f3_a, m_f3_b);
        shift_hist<3>(f3_out, m_f3_yh);

        double f3_plus_noise = f3_out + noise_direct;

        shift_hist<4>(f3_plus_noise, m_f4_xh);
        double f4_out = apply_filter<4, 4>(m_f4_xh, m_f4_yh, m_f4_a, m_f4_b);
        shift_hist<3>(f4_out, m_f4_yh);

        double closure_atten = (7 ^ (m_closure >> 2)) / 7.0;
        double closure_out = f4_out * closure_atten;

        shift_hist<2>(closure_out, m_fx_xh);
        double fx_out = apply_filter<2, 2>(m_fx_xh, m_fx_yh, m_fx_a, m_fx_b);
        shift_hist<1>(fx_out, m_fx_yh);

        return fx_out * 0.35;
    }
};
