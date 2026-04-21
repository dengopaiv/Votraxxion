// Votrax SC-01A chip-level emulation — C++ core.
// Faithfully reproduces the MAME votrax.cpp analog signal path.
#pragma once

#include "filters.h"
#include "rom_data.h"

// 9-level glottal waveform from MAME
static constexpr double GLOTTAL[9] = {
    0.0, -4.0/7.0, 1.0, 6.0/7.0, 5.0/7.0, 4.0/7.0, 3.0/7.0, 2.0/7.0, 1.0/7.0
};

class VotraxSC01ACore {
public:
    // `master_clock` is the SC-01A external clock in Hz. Nominal 720 000; the
    // datasheet endorses varying it for sound-design effects (Figures 6/7).
    // Smaller → slower/lower-pitched, larger → faster/higher-pitched.
    // `fx_fudge` scales the final-stage lowpass cutoff. 150/4000 (default)
    // matches MAME's observed-from-recordings behavior; 1.0 is "as-schematic".
    // `closure_strength` scales how deeply plosive closures attenuate the output.
    // 1.0 (default) reproduces MAME's curve; 0.0 disables the closure dip so
    // plosives lose their punch; values >1.0 exaggerate the closure effect.
    explicit VotraxSC01ACore(double master_clock = DEFAULT_MASTER_CLOCK,
                             double fx_fudge = 150.0 / 4000.0,
                             double closure_strength = 1.0)
        : m_master_clock(master_clock),
          m_sclock(sclock_from_master(master_clock)),
          m_cclock(cclock_from_master(master_clock)),
          m_fx_fudge(fx_fudge),
          m_closure_strength(closure_strength)
    {
        reset();
    }

    double master_clock() const { return m_master_clock; }
    double sclock() const { return m_sclock; }
    double cclock() const { return m_cclock; }
    double fx_fudge() const { return m_fx_fudge; }
    double closure_strength() const { return m_closure_strength; }

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

    // Commit a phoneme with explicit parameter overrides, bypassing the ROM
    // lookup. All 12 fields become user-controlled: this turns the chip into a
    // formant instrument driven by arbitrary parameters. m_phone is still
    // stored (for reporting) but not used to fetch params.
    void phone_commit_override(int phone, int inflection, const PhonemeParams& params) {
        m_phone = phone & 0x3F;
        m_inflection = inflection & 0x03;
        m_rom = params;
        m_phonetick = 0;
        m_ticks = 0;
        if (m_rom.cld == 0)
            m_cur_closure = m_rom.closure;
    }

    // Read-only accessor: get the ROM-decoded parameters for a given phoneme
    // code. Useful for UIs that want to show "defaults" alongside user overrides.
    static PhonemeParams rom_params(int phone) {
        return ROM_DATA[phone & 0x3F];
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

private:
    // --- Clock and tunable-curve configuration (set at construction) ---
    double m_master_clock;
    double m_sclock;
    double m_cclock;
    double m_fx_fudge;
    double m_closure_strength;

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
        build_standard_filter(m_f4_a, m_f4_b, m_sclock, m_cclock,
            0, 28810, 1165, 21457, 8558, 7289);
        build_lowpass_filter(m_fx_a, m_fx_b, m_sclock, m_cclock,
            1122, 23131, m_fx_fudge);
        build_noise_shaper_filter(m_ns_a, m_ns_b, m_sclock, m_cclock,
            15500, 14854, 8450, 9523, 14083);
    }

    void build_variable_filters() {
        double f1_caps[] = {2546, 4973, 9861, 19724};
        double f1_c3 = 2280 + bits_to_caps(m_filt_f1, f1_caps, 4);
        build_standard_filter(m_f1_a, m_f1_b, m_sclock, m_cclock,
            11247, 11797, 949, 52067, f1_c3, 166272);

        double f2q_caps[] = {1390, 2965, 5875, 11297};
        double f2_caps[] = {833, 1663, 3164, 6327, 12654};
        double f2v_c2t = 829 + bits_to_caps(m_filt_f2q, f2q_caps, 4);
        double f2v_c3 = 2352 + bits_to_caps(m_filt_f2, f2_caps, 5);
        build_standard_filter(m_f2v_a, m_f2v_b, m_sclock, m_cclock,
            24840, 29154, f2v_c2t, 38180, f2v_c3, 34270);

        double f3_caps[] = {2226, 4485, 9056, 18111};
        double f3_c3 = 8480 + bits_to_caps(m_filt_f3, f3_caps, 4);
        build_standard_filter(m_f3_a, m_f3_b, m_sclock, m_cclock,
            0, 17594, 868, 18828, f3_c3, 50019);

        build_injection_filter(m_f2n_a, m_f2n_b, m_sclock, m_cclock,
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
        if (m_pitch == pitch_target)
            m_pitch = 0;

        if ((m_pitch & 0xF9) == 0x08)
            commit_filters();

        // Noise LFSR
        bool inp = (true || m_filt_fa) && m_cur_noise && (m_noise != 0x7FFF);
        m_noise = ((m_noise << 1) & 0x7FFE) | (inp ? 1 : 0);
        m_cur_noise = !(((m_noise >> 14) ^ (m_noise >> 13)) & 1);
    }

    double analog_calc() {
        // Original 9-level MAME glottal waveform
        int glot_idx = m_pitch >> 3;
        double glottal = (glot_idx < 9) ? GLOTTAL[glot_idx] : 0.0;

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

        // MAME closure curve, scaled by closure_strength: 1.0 = authentic,
        // 0.0 = no closure dip, >1.0 = exaggerated.
        double mame_atten = (7 ^ (m_closure >> 2)) / 7.0;
        double closure_atten = 1.0 - m_closure_strength * (1.0 - mame_atten);
        double closure_out = f4_out * closure_atten;

        shift_hist<2>(closure_out, m_fx_xh);
        double fx_out = apply_filter<2, 2>(m_fx_xh, m_fx_yh, m_fx_a, m_fx_b);
        shift_hist<1>(fx_out, m_fx_yh);

        return fx_out * 0.35;
    }
};
