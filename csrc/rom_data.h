// Votrax SC-01A ROM data — pre-decoded phoneme parameters.
// Generated from pyvotrax/rom.py RAW_ROM data.
#pragma once

#include <cstdint>
#include <array>

struct PhonemeParams {
    int f1;       // 4-bit filter 1 frequency
    int va;       // 4-bit voice amplitude
    int f2;       // 4-bit filter 2 frequency
    int fc;       // 4-bit fricative/noise control
    int f2q;      // 4-bit filter 2 Q factor
    int f3;       // 4-bit filter 3 frequency
    int fa;       // 4-bit fricative amplitude
    int cld;      // 4-bit closure delay
    int vd;       // 4-bit voice delay
    int closure;  // 1-bit closure flag
    int duration; // 7-bit duration
    bool pause;   // true if pause phoneme
};

// Raw ROM data: rom[64][2] from rom.cc
static constexpr uint32_t RAW_ROM_W0[64] = {
    0x361, 0x161, 0x9A1, 0x0E0, 0x0FB, 0x161, 0x7A1, 0x463,
    0x161, 0xB61, 0xA61, 0x9A1, 0x7A3, 0xA61, 0x173, 0x163,
    0x163, 0x9A2, 0x163, 0x8A0, 0x9A0, 0x8A1, 0x7A1, 0xA21,
    0x7A1, 0xA72, 0x0E8, 0x162, 0x173, 0x7A2, 0xB7C, 0x468,
    0xA21, 0x561, 0xA61, 0x0E3, 0xCC1, 0x7B2, 0xA21, 0xA21,
    0xA21, 0x7A1, 0x172, 0x463, 0xA20, 0xA66, 0xA20, 0x7A0,
    0x461, 0x163, 0x7A1, 0xA21, 0xA61, 0x9A1, 0x366, 0x461,
    0xA63, 0x168, 0x8A1, 0xA21, 0x9A1, 0xCC1, 0xA23, 0x0F0,
};

static constexpr uint32_t RAW_ROM_W1[64] = {
    0x74688127, 0xD4688127, 0xC4688127, 0xF0A050A4,
    0x610316E8, 0x64C9C1A6, 0x34C9C1A6, 0xF3CB546C,
    0xC4E940A3, 0x806191A6, 0x906191A6, 0x906191A6,
    0x66A58832, 0xE6241936, 0x90E19122, 0xF7D36428,
    0xFB8B546C, 0xFB8B546C, 0x9CD15860, 0x706980A3,
    0xD4084B36, 0x84E940A3, 0x30498123, 0x20498123,
    0xF409D0A2, 0x1123642C, 0xDB7B342C, 0xFD2204AC,
    0xE041C126, 0x65832CA8, 0x00E89126, 0x489132E0,
    0x84C9C1A6, 0x7069D326, 0x64A01226, 0x548981A3,
    0x84E940A3, 0x631324A8, 0x84E8C1A2, 0x806191A6,
    0x80E8C122, 0x64015326, 0xE81132E0, 0x54084382,
    0x7049D326, 0x1460C122, 0x74E880A7, 0x74E880A7,
    0x606980A3, 0x548981A3, 0xE48981A3, 0xB48981A3,
    0x34E8C1A2, 0x80E8C1A2, 0x106083A2, 0x90E8C122,
    0x88E15220, 0x183800A4, 0x2448C382, 0x94688127,
    0x9049D326, 0xB06980A3, 0x00A050A4, 0x30A058A4,
};

static inline int extract_param(uint32_t word1, int slot) {
    uint32_t base = word1 >> slot;
    return (((base & 0x000001) ? 8 : 0) |
            ((base & 0x000080) ? 4 : 0) |
            ((base & 0x004000) ? 2 : 0) |
            ((base & 0x200000) ? 1 : 0));
}

static inline int extract_clvd(uint32_t word0, uint32_t word1, int slot) {
    uint32_t base = (word1 >> 28) | (word0 << 4);
    if (slot == 6) base >>= 1;
    return (((base & 0x01) ? 1 : 0) |
            ((base & 0x04) ? 2 : 0) |
            ((base & 0x10) ? 4 : 0) |
            ((base & 0x40) ? 8 : 0));
}

static inline PhonemeParams decode_phoneme(int index) {
    uint32_t w0 = RAW_ROM_W0[index];
    uint32_t w1 = RAW_ROM_W1[index];

    int duration = (((w0 & 0x020) ? 0x40 : 0) |
                    ((w0 & 0x040) ? 0x20 : 0) |
                    ((w0 & 0x080) ? 0x10 : 0) |
                    ((w0 & 0x100) ? 0x08 : 0) |
                    ((w0 & 0x200) ? 0x04 : 0) |
                    ((w0 & 0x400) ? 0x02 : 0) |
                    ((w0 & 0x800) ? 0x01 : 0)) ^ 0x7F;

    return PhonemeParams{
        extract_param(w1, 0),      // f1
        extract_param(w1, 1),      // va
        extract_param(w1, 2),      // f2
        extract_param(w1, 3),      // fc
        extract_param(w1, 4),      // f2q
        extract_param(w1, 5),      // f3
        extract_param(w1, 6),      // fa
        extract_clvd(w0, w1, 0),   // cld
        extract_clvd(w0, w1, 6),   // vd
        (w0 & 0x10) ? 1 : 0,      // closure
        duration,                   // duration
        (index == 0x03 || index == 0x3E)  // pause
    };
}

// Pre-decoded ROM data, initialized at startup
static inline std::array<PhonemeParams, 64> build_rom_data() {
    std::array<PhonemeParams, 64> data;
    for (int i = 0; i < 64; i++) {
        data[i] = decode_phoneme(i);
    }
    return data;
}

static const std::array<PhonemeParams, 64> ROM_DATA = build_rom_data();
