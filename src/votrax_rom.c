/* Votrax SC-01 / SC-01-A mask ROM -- the raw words and their decode.
 *
 * See votrax_rom.h for provenance.  The two words per phone are the die's own
 * bits; everything the chip knows about a phone is extracted from them by the
 * three functions at the bottom of this file, which reproduce the extraction
 * in reference/gate-sim/rom.cc bit for bit.  The bit orders look arbitrary
 * because they are: they are wiring, not encoding.
 */

#include "votrax_rom.h"

/* The SC-01-A mask. */
static const uint32_t RAW_ROM_W0[64] = {
    0x361, 0x161, 0x9A1, 0x0E0, 0x0FB, 0x161, 0x7A1, 0x463,
    0x161, 0xB61, 0xA61, 0x9A1, 0x7A3, 0xA61, 0x173, 0x163,
    0x163, 0x9A2, 0x163, 0x8A0, 0x9A0, 0x8A1, 0x7A1, 0xA21,
    0x7A1, 0xA72, 0x0E8, 0x162, 0x173, 0x7A2, 0xB7C, 0x468,
    0xA21, 0x561, 0xA61, 0x0E3, 0xCC1, 0x7B2, 0xA21, 0xA21,
    0xA21, 0x7A1, 0x172, 0x463, 0xA20, 0xA66, 0xA20, 0x7A0,
    0x461, 0x163, 0x7A1, 0xA21, 0xA61, 0x9A1, 0x366, 0x461,
    0xA63, 0x168, 0x8A1, 0xA21, 0x9A1, 0xCC1, 0xA23, 0x0F0,
};

static const uint32_t RAW_ROM_W1[64] = {
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

/* The 1980 SC-01 mask differs from the SC-01-A in exactly twelve rows, and
 * every difference is confined to one field: the voice amplitude (va) of the
 * open vowels.  The original ran all twelve at full scale (va = 15); the -A
 * revision pulled them down to 9, 11 or 14.  word0 is identical throughout, so
 * only word1 needs overriding.  Audibly the SC-01 is the louder, more strident
 * voice -- roughly +29% RMS and +48% peak on ordinary speech. */
typedef struct { uint8_t phone; uint32_t w1; } vx_mask_delta;

static const vx_mask_delta SC01_W1_DELTAS[12] = {
    { 0x08, 0xC4E9C1A3 },  // AH2   va 9  -> 15
    { 0x13, 0x706981A3 },  // AW1   va 11 -> 15
    { 0x15, 0x84E9C1A3 },  // AH1   va 9  -> 15
    { 0x23, 0x54C981A3 },  // UH3   va 14 -> 15
    { 0x24, 0x84E9C1A3 },  // AH    va 9  -> 15
    { 0x2E, 0x74E881A7 },  // AE    va 11 -> 15
    { 0x2F, 0x74E881A7 },  // AE1   va 11 -> 15
    { 0x30, 0x606981A3 },  // AW2   va 11 -> 15
    { 0x31, 0x54C981A3 },  // UH2   va 14 -> 15
    { 0x32, 0xE4C981A3 },  // UH1   va 14 -> 15
    { 0x33, 0xB4C981A3 },  // UH    va 14 -> 15
    { 0x3D, 0xB06981A3 },  // AW    va 11 -> 15
};

/* --- decode ------------------------------------------------------------- */

/* One 4-bit parameter out of word1.  The four bits of a parameter are 7, 14
 * and 7 bits apart, and the slot selects which of the seven parameters. */
static int extract_param(uint32_t word1, int slot)
{
    uint32_t base = word1 >> slot;
    return (((base & 0x000001u) ? 8 : 0) |
            ((base & 0x000080u) ? 4 : 0) |
            ((base & 0x004000u) ? 2 : 0) |
            ((base & 0x200000u) ? 1 : 0));
}

/* The closure-delay and voice-delay fields straddle the two words. */
static int extract_clvd(uint32_t word0, uint32_t word1, int slot)
{
    uint32_t base = (word1 >> 28) | (word0 << 4);
    if (slot == 6)
        base >>= 1;
    return (((base & 0x01u) ? 1 : 0) |
            ((base & 0x04u) ? 2 : 0) |
            ((base & 0x10u) ? 4 : 0) |
            ((base & 0x40u) ? 8 : 0));
}

/* word1 for `index` under `rev`: the -A table, with the 1980 overrides. */
static uint32_t raw_word1(int index, vx_mask_revision rev)
{
    int i;
    if (rev == VX_ROM_SC01) {
        for (i = 0; i < 12; i++)
            if (SC01_W1_DELTAS[i].phone == (uint8_t)index)
                return SC01_W1_DELTAS[i].w1;
    }
    return RAW_ROM_W1[index];
}

static vx_phoneme decode_phoneme(int index, vx_mask_revision rev)
{
    uint32_t w0 = RAW_ROM_W0[index];
    uint32_t w1 = raw_word1(index, rev);
    vx_phoneme p;

    /* Duration is stored bit-reversed and inverted. */
    p.duration = (((w0 & 0x020u) ? 0x40 : 0) |
                  ((w0 & 0x040u) ? 0x20 : 0) |
                  ((w0 & 0x080u) ? 0x10 : 0) |
                  ((w0 & 0x100u) ? 0x08 : 0) |
                  ((w0 & 0x200u) ? 0x04 : 0) |
                  ((w0 & 0x400u) ? 0x02 : 0) |
                  ((w0 & 0x800u) ? 0x01 : 0)) ^ 0x7F;

    p.f1  = extract_param(w1, 0);
    p.va  = extract_param(w1, 1);
    p.f2  = extract_param(w1, 2);
    p.fc  = extract_param(w1, 3);
    p.f2q = extract_param(w1, 4);
    p.f3  = extract_param(w1, 5);
    p.fa  = extract_param(w1, 6);
    p.cld = extract_clvd(w0, w1, 0);
    p.vd  = extract_clvd(w0, w1, 6);
    p.closure = (w0 & 0x10u) ? 1 : 0;
    p.pause = (index == 0x03 || index == 0x3E);
    return p;
}

/* --- the decoded tables -------------------------------------------------- */

static vx_phoneme ROM_SC01A[64];
static vx_phoneme ROM_SC01[64];
static int rom_decoded;

const vx_phoneme *vx_rom_table(vx_mask_revision rev)
{
    if (!rom_decoded) {
        int i;
        for (i = 0; i < 64; i++) {
            ROM_SC01A[i] = decode_phoneme(i, VX_ROM_SC01A);
            ROM_SC01[i]  = decode_phoneme(i, VX_ROM_SC01);
        }
        /* Set last: a racing reader either sees 0 and decodes again into the
         * same values, or sees 1 with both tables already written. */
        rom_decoded = 1;
    }
    return rev == VX_ROM_SC01 ? ROM_SC01 : ROM_SC01A;
}

vx_phoneme vx_rom_phoneme(int phone, vx_mask_revision rev)
{
    return vx_rom_table(rev)[phone & 0x3F];
}
