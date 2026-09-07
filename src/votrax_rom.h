/* Votrax SC-01 / SC-01-A internal mask ROM -- decoded phoneme parameters.
 *
 * The SC-01 keeps its 64 phoneme definitions in an on-die mask ROM: 64 rows of
 * a 12-bit word0 and a 32-bit word1.  There is nothing to load at runtime --
 * the silicon's ROM is immutable, so the tables in votrax_rom.c ARE the chip.
 *
 * Provenance: the SC-01-A words were transcribed from the die by Olivier
 * Galibert (reference/gate-sim/rom.cc); they have since been verified
 * byte-for-byte against the dumped 512-byte mask ROM (CRC32 fc416227, SHA1
 * 1d6da90b1807a01b5e186ef08476119a862b5e6d).  The 1980 SC-01 deltas come from
 * that mask's dump (CRC32 528d1c57, SHA1 268b5884dce04e49e2376df3e2dc82e852b7
 * 08c1).  See docs/tech-overview.md, Part 1, "The two mask revisions".
 */
#ifndef VOTRAX_ROM_H
#define VOTRAX_ROM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The twelve parameters the chip reads for each phone. */
typedef struct {
    int f1;       /* 4-bit filter 1 frequency        */
    int va;       /* 4-bit voice amplitude           */
    int f2;       /* 4-bit filter 2 frequency        */
    int fc;       /* 4-bit fricative/noise control   */
    int f2q;      /* 4-bit filter 2 Q factor         */
    int f3;       /* 4-bit filter 3 frequency        */
    int fa;       /* 4-bit fricative amplitude       */
    int cld;      /* 4-bit closure delay             */
    int vd;       /* 4-bit voice delay               */
    int closure;  /* 1-bit closure flag              */
    int duration; /* 7-bit duration                  */
    int pause;    /* nonzero if a pause phone        */
} vx_phoneme;

/* Which silicon revision to speak with.  Both are real production silicon:
 * the SC-01 is the 1980 part, the SC-01-A the later revision that most
 * surviving hardware carries. */
typedef enum {
    VX_ROM_SC01A = 0,
    VX_ROM_SC01  = 1
} vx_mask_revision;

/* The 64 decoded phones for one revision.  The pointer is to storage that
 * lives for the life of the program; do not free it, and do not write through
 * it -- vx_core keeps its own copy of the row it is voicing precisely so that
 * parameter overrides cannot reach back into the ROM.
 *
 * Decoding happens once, on the first call.  Two threads racing here compute
 * identical bytes into the same storage, so the race is benign; there is no
 * partially-valid state a reader could observe. */
const vx_phoneme *vx_rom_table(vx_mask_revision rev);

/* One row, for callers that want a phone's defaults without the whole table. */
vx_phoneme vx_rom_phoneme(int phone, vx_mask_revision rev);

#ifdef __cplusplus
}
#endif

#endif /* VOTRAX_ROM_H */
