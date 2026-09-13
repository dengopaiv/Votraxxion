/* votrax-say -- the SC-01 synthesizer on the command line, text or phones to WAV.
 *
 *   votrax-say "Hello world."                  text, to votrax.wav
 *   votrax-say -o hi.wav --phones "H EH1 L O1 U1"
 *   votrax-say --spell "NVDA"                  read it letter by letter
 *   echo "From a pipe." | votrax-say -o pipe.wav
 *   votrax-say --table -o table.wav            all 64 phones, timed on stdout
 *   votrax-say --names                         the phone table, no audio
 *   votrax-say --print "Hello world."          the phones text would make
 *
 * Plain C11 over the public API in src/votrax.h, compiled in with the six
 * engine sources -- no DLL beside it, no data file, nothing platform specific.
 * The WAV header is written byte by byte, little-endian, so the output is the
 * same file on any host.
 *
 * The modes -- phone strings, the whole 64-phone table, the name list -- follow
 * Tamas Geczy's say01 probe in votraxsc01-nvda; the code is this repository's.
 *
 * The phone grammar is the GUI's: NAME[:LEVEL] tokens separated by spaces or
 * commas, names from the datasheet (case-insensitive), levels 0-3 with the
 * neutral level when omitted.  --print emits exactly that grammar, so its
 * output can be edited and fed back through --phones. */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "votrax.h"

#define PROGRAM "votrax-say"

typedef enum { MODE_TEXT, MODE_PHONES, MODE_SPELL, MODE_TABLE, MODE_NAMES } input_mode;

typedef struct {
    input_mode   mode;
    const char  *input;        /* NULL: read stdin */
    const char  *wav_path;
    int          mask;
    unsigned int clock_hz;
    double       speed;
    int          inflection;
    int          flat;
    int          print_only;
    int          output_stage;
    double       volume;
} options;

static void usage(FILE *to)
{
    fprintf(to,
        "usage: " PROGRAM " [options] [TEXT]\n"
        "\n"
        "Speak TEXT (or standard input) through the Votrax SC-01 and write a WAV.\n"
        "\n"
        "input, one of:\n"
        "  TEXT                  English text (the default mode)\n"
        "  --phones PHONES       datasheet phone names, e.g. \"H EH1 L O1 U1\" or \"AY:3 M\"\n"
        "  --spell TEXT          read TEXT character by character\n"
        "  --table               every phone 00-3F in order, each followed by PA1\n"
        "  --names               print the 64 phone codes and names, and stop\n"
        "\n"
        "voice:\n"
        "  --mask sc01a|sc01     mask revision (default sc01a)\n"
        "  --clock HZ            master clock, 100000-4000000 (default 720000)\n"
        "  --knob P              clock from the data sheet's voice knob, 0-1\n"
        "                        (0 fast and high, 1 slow and low, 720 kHz near 0.6)\n"
        "  --rc OHMS,FARADS      clock from the MCRC parts, f = 1.25/RC\n"
        "  --speed X             tempo at constant pitch, 0.1-10 (default 1)\n"
        "  --inflection N        base pitch level 0-3 (default 1, neutral)\n"
        "  --flat                no sentence contour on text\n"
        "  --output-stage S      chip (the AO pin, default) or figure8 (the\n"
        "                        data sheet's LM386 amplifier into a speaker)\n"
        "  --volume P            figure8's volume control, 0-1 (default 1, full;\n"
        "                        the 1980 mask's loudest vowels clip above ~0.93)\n"
        "\n"
        "output:\n"
        "  -o, --output FILE     WAV path (default votrax.wav)\n"
        "  --print               print the phones instead of writing audio\n"
        "  -h, --help            this text\n");
}

static int fail(const char *message, const char *detail)
{
    fprintf(stderr, PROGRAM ": %s%s%s\n", message,
            detail ? ": " : "", detail ? detail : "");
    return 2;
}

/* --------------------------------------------------------------- input --- */

static char *read_stream(FILE *f)
{
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf)
        return NULL;
    for (;;) {
        size_t got;
        if (len + 1 >= cap) {
            char *bigger = (char *)realloc(buf, cap * 2);
            if (!bigger) {
                free(buf);
                return NULL;
            }
            buf = bigger;
            cap *= 2;
        }
        got = fread(buf + len, 1, cap - len - 1, f);
        len += got;
        if (got == 0)
            break;
    }
    buf[len] = '\0';
    return buf;
}

/* NAME[:LEVEL] tokens to packed phones.  Returns the count, or -1 with a
 * message naming the token that was wrong. */
static int parse_phones(const char *s, unsigned char **out)
{
    unsigned char *phones = NULL;
    int count = 0, cap = 0, index = 0;

    while (*s) {
        char name[16];
        size_t len = 0;
        int level = VX_NEUTRAL_INFLECTION, code, truncated = 0;

        while (*s && (isspace((unsigned char)*s) || *s == ','))
            s++;
        if (!*s)
            break;
        index++;
        while (*s && !isspace((unsigned char)*s) && *s != ',') {
            if (len + 1 < sizeof name)
                name[len++] = (char)toupper((unsigned char)*s);
            else
                truncated = 1;
            s++;
        }
        name[len] = '\0';

        if (!truncated && len >= 3 && name[len - 2] == ':' &&
            name[len - 1] >= '0' && name[len - 1] <= '3') {
            level = name[len - 1] - '0';
            name[len - 2] = '\0';
        }
        code = truncated ? -1 : vx_phone_by_name(name);
        if (code < 0) {
            fprintf(stderr, PROGRAM ": phone %d is not a phone name: \"%s\""
                    " (--names lists them)\n", index, name);
            free(phones);
            return -1;
        }
        if (count == cap) {
            int grow = cap ? cap * 2 : 256;
            unsigned char *bigger = (unsigned char *)realloc(phones, (size_t)grow);
            if (!bigger) {
                free(phones);
                return -1;
            }
            phones = bigger;
            cap = grow;
        }
        phones[count++] = VX_PACK(code, level);
    }
    *out = phones;
    return count;
}

/* Text through the front end, sized by asking first: ttv_* report the count
 * they would have produced, so one retry always fits. */
static int translate(const char *text, int (*fn)(const char *, unsigned char *, int),
                     unsigned char **out)
{
    int n = fn(text, NULL, 0);
    unsigned char *phones;
    if (n <= 0) {
        *out = NULL;
        return 0;
    }
    phones = (unsigned char *)malloc((size_t)n);
    if (!phones)
        return -1;
    fn(text, phones, n);
    *out = phones;
    return n;
}

/* -------------------------------------------------------------- output --- */

static void put_u16(unsigned char *p, unsigned v)
{
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
}

static void put_u32(unsigned char *p, unsigned long v)
{
    put_u16(p, (unsigned)(v & 0xFFFF));
    put_u16(p + 2, (unsigned)((v >> 16) & 0xFFFF));
}

/* 16-bit mono PCM at the chip's own rate -- clock / 18, so 40 kHz at the
 * datasheet clock.  A non-integer rate (clock not a multiple of 18) is
 * rounded in the header, a drift under 0.005% at any clock the tool allows. */
static int write_wav(const char *path, const int16_t *pcm, long samples, unsigned long rate)
{
    unsigned char h[44];
    unsigned long data = (unsigned long)samples * 2;
    unsigned char *bytes;
    long i;
    FILE *f;
    int ok;

    memcpy(h, "RIFF", 4);
    put_u32(h + 4, 36 + data);
    memcpy(h + 8, "WAVEfmt ", 8);
    put_u32(h + 16, 16);
    put_u16(h + 20, 1);            /* PCM */
    put_u16(h + 22, 1);            /* mono */
    put_u32(h + 24, rate);
    put_u32(h + 28, rate * 2);
    put_u16(h + 32, 2);
    put_u16(h + 34, 16);
    memcpy(h + 36, "data", 4);
    put_u32(h + 40, data);

    bytes = (unsigned char *)malloc(data ? data : 1);
    if (!bytes)
        return 0;
    for (i = 0; i < samples; i++)
        put_u16(bytes + 2 * i, (unsigned)(uint16_t)pcm[i]);

    f = fopen(path, "wb");
    if (!f) {
        free(bytes);
        return 0;
    }
    ok = fwrite(h, 1, sizeof h, f) == sizeof h &&
         fwrite(bytes, 1, data, f) == data;
    ok = (fclose(f) == 0) && ok;
    free(bytes);
    return ok;
}

static void print_phones(const unsigned char *phones, int count)
{
    int i;
    for (i = 0; i < count; i++) {
        int level = VX_INFLECTION(phones[i]);
        printf("%s%s", i ? " " : "", vx_phone_name(VX_PHONE(phones[i])));
        if (level != VX_NEUTRAL_INFLECTION)
            printf(":%d", level);
    }
    printf("\n");
}

/* ----------------------------------------------------------- synthesis --- */

typedef struct {
    int16_t *pcm;
    long     count, cap;
} pcm_buf;

static int render_into(vx_chip *chip, pcm_buf *b, int n)
{
    if (b->count + n > b->cap) {
        long grow = b->cap ? b->cap : 65536;
        int16_t *bigger;
        while (grow < b->count + n)
            grow *= 2;
        bigger = (int16_t *)realloc(b->pcm, (size_t)grow * sizeof *bigger);
        if (!bigger)
            return 0;
        b->pcm = bigger;
        b->cap = grow;
    }
    b->count += vx_render(chip, b->pcm + b->count, n);
    return 1;
}

/* Through the scheduler, topping the queue up as it drains -- it holds
 * VX_QUEUE_CAPACITY and drops the rest -- then a quarter-second tail, because
 * an empty queue is not silence: the chip is still voicing the last phone. */
static int speak(vx_chip *chip, const unsigned char *phones, int count, pcm_buf *b)
{
    const int block = 1024;
    int queued = 0;
    int rate = (int)(vx_sample_rate(chip) + 0.5);
    long limit = (long)rate * 3600;    /* an hour: a stop, not a hang */

    for (;;) {
        if (queued < count) {
            int room = VX_QUEUE_CAPACITY - vx_pending(chip);
            if (room > 0) {
                int take = count - queued < room ? count - queued : room;
                vx_speak(chip, phones + queued, take);
                queued += take;
            }
        } else if (vx_pending(chip) == 0) {
            break;
        }
        if (b->count >= limit)
            break;
        if (!render_into(chip, b, block))
            return 0;
    }
    return render_into(chip, b, rate / 4);
}

/* Hold a phone for its natural length over `speed`, written directly rather
 * than queued, so the caller knows the sample it starts on.  (Through the
 * queue, vx_pending reaches zero when the last phone is committed, not when it
 * ends, and a start time read from that is one phone early.) */
static int hold(vx_chip *chip, pcm_buf *b, unsigned char phone, double speed)
{
    int n = (int)(vx_phone_samples(chip, phone) / speed + 0.5);
    vx_write(chip, phone);
    return render_into(chip, b, n > 0 ? n : 1);
}

/* Every phone, each followed by PA1 so a listener or an editor can find the
 * boundaries, with its start time and length on stdout. */
static int table(vx_chip *chip, pcm_buf *b, double speed)
{
    double rate = vx_sample_rate(chip);
    int code;
    printf("code  name  start_s  length_ms\n");
    for (code = 0; code < 64; code++) {
        printf("%02X    %-4s  %7.3f  %9.1f\n", code, vx_phone_name(code),
               (double)b->count / rate,
               1000.0 * vx_phone_samples(chip, (unsigned char)code) / speed / rate);
        if (!hold(chip, b, (unsigned char)code, speed) ||
            !hold(chip, b, 0x3E, speed))           /* PA1 */
            return 0;
    }
    vx_write(chip, 0x3F);                          /* STOP */
    return render_into(chip, b, (int)(rate / 4));
}

/* ---------------------------------------------------------------- main --- */

static int parse_double(const char *s, double lo, double hi, double *out)
{
    char *end;
    double v = strtod(s, &end);
    if (end == s || *end || v < lo || v > hi)
        return 0;
    *out = v;
    return 1;
}

int main(int argc, char **argv)
{
    options o;
    int i, count = 0, status = 0;
    unsigned char *phones = NULL;
    char *owned_input = NULL;
    vx_chip *chip;
    pcm_buf b = { NULL, 0, 0 };

    memset(&o, 0, sizeof o);
    o.mode = MODE_TEXT;
    o.wav_path = "votrax.wav";
    o.mask = VX_MASK_SC01A;
    o.clock_hz = VX_BASE_CLOCK;
    o.output_stage = VX_OUTPUT_CHIP;
    o.volume = 1.0;
    o.speed = 1.0;
    o.inflection = VX_NEUTRAL_INFLECTION;

    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *next = i + 1 < argc ? argv[i + 1] : NULL;
        double v;

        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(a, "--phones") || !strcmp(a, "--spell")) {
            if (!next) return fail("missing value for", a);
            o.mode = a[2] == 'p' ? MODE_PHONES : MODE_SPELL;
            o.input = argv[++i];
        } else if (!strcmp(a, "--table")) {
            o.mode = MODE_TABLE;
        } else if (!strcmp(a, "--names")) {
            o.mode = MODE_NAMES;
        } else if (!strcmp(a, "--mask")) {
            if (!next) return fail("missing value for", a);
            if (!strcmp(next, "sc01a") || !strcmp(next, "SC01A")) o.mask = VX_MASK_SC01A;
            else if (!strcmp(next, "sc01") || !strcmp(next, "SC01")) o.mask = VX_MASK_SC01;
            else return fail("--mask is sc01a or sc01, not", next);
            i++;
        } else if (!strcmp(a, "--clock")) {
            if (!next || !parse_double(next, 100000, 4000000, &v))
                return fail("--clock wants 100000-4000000 Hz", next);
            o.clock_hz = (unsigned int)(v + 0.5);
            i++;
        } else if (!strcmp(a, "--knob")) {
            if (!next || !parse_double(next, 0, 1, &v))
                return fail("--knob wants a position from 0 to 1", next);
            o.clock_hz = vx_clock_from_knob(v);
            i++;
        } else if (!strcmp(a, "--rc")) {
            char *comma, *end;
            double ohms, farads;
            if (!next)
                return fail("missing value for", a);
            ohms = strtod(next, &comma);
            if (comma == next || *comma != ',')
                return fail("--rc wants OHMS,FARADS, e.g. 6800,120e-12", next);
            farads = strtod(comma + 1, &end);
            if (end == comma + 1 || *end)
                return fail("--rc wants OHMS,FARADS, e.g. 6800,120e-12", next);
            o.clock_hz = vx_clock_from_rc(ohms, farads);
            if (o.clock_hz < 100000 || o.clock_hz > 4000000)
                return fail("--rc gives a clock outside 100000-4000000 Hz", next);
            i++;
        } else if (!strcmp(a, "--volume")) {
            if (!next || !parse_double(next, 0, 1, &v))
                return fail("--volume wants a position from 0 to 1", next);
            o.volume = v;
            i++;
        } else if (!strcmp(a, "--output-stage")) {
            if (!next) return fail("missing value for", a);
            if (!strcmp(next, "chip")) o.output_stage = VX_OUTPUT_CHIP;
            else if (!strcmp(next, "figure8")) o.output_stage = VX_OUTPUT_FIGURE8;
            else return fail("--output-stage is chip or figure8, not", next);
            i++;
        } else if (!strcmp(a, "--speed")) {
            if (!next || !parse_double(next, 0.1, 10.0, &v))
                return fail("--speed wants 0.1-10", next);
            o.speed = v;
            i++;
        } else if (!strcmp(a, "--inflection")) {
            if (!next || !parse_double(next, 0, 3, &v) || v != (int)v)
                return fail("--inflection wants 0, 1, 2 or 3", next);
            o.inflection = (int)v;
            i++;
        } else if (!strcmp(a, "--flat")) {
            o.flat = 1;
        } else if (!strcmp(a, "-o") || !strcmp(a, "--output")) {
            if (!next) return fail("missing value for", a);
            o.wav_path = argv[++i];
        } else if (!strcmp(a, "--print")) {
            o.print_only = 1;
        } else if (a[0] == '-' && a[1] != '\0') {
            usage(stderr);
            return fail("unknown option", a);
        } else {
            if (o.input && o.mode == MODE_TEXT)
                return fail("give the text as one argument (quote it)", a);
            o.input = a;
        }
    }

    if (o.mode == MODE_NAMES) {
        for (i = 0; i < 64; i++)
            printf("%02X  %s\n", i, vx_phone_name(i));
        return 0;
    }

    if (o.mode != MODE_TABLE && !o.input) {
        owned_input = read_stream(stdin);
        if (!owned_input)
            return fail("could not read standard input", NULL);
        o.input = owned_input;
    }

    switch (o.mode) {
    case MODE_PHONES: count = parse_phones(o.input, &phones); break;
    case MODE_SPELL:  count = translate(o.input, ttv_spell, &phones); break;
    case MODE_TEXT:
        count = translate(o.input, o.flat ? ttv_translate_flat : ttv_translate,
                          &phones);
        break;
    default: break;
    }
    free(owned_input);
    if (count < 0) {
        free(phones);
        return 2;
    }
    if (o.mode != MODE_TABLE && count == 0) {
        free(phones);
        return fail("nothing to speak", NULL);
    }

    if (o.print_only) {
        if (o.mode == MODE_TABLE)
            return fail("--print has nothing to print for --table", NULL);
        print_phones(phones, count);
        free(phones);
        return 0;
    }

    chip = vx_create(o.mask, o.clock_hz);
    if (!chip) {
        free(phones);
        return fail("out of memory", NULL);
    }
    vx_set_speed(chip, o.speed);
    vx_inflection(chip, (unsigned char)o.inflection);
    vx_set_output(chip, o.output_stage);
    vx_set_output_volume(chip, o.volume);

    if (!(o.mode == MODE_TABLE ? table(chip, &b, o.speed)
                               : speak(chip, phones, count, &b)))
        status = fail("out of memory", NULL);
    else if (!write_wav(o.wav_path, b.pcm, b.count,
                        (unsigned long)(vx_sample_rate(chip) + 0.5)))
        status = fail("could not write", o.wav_path);
    else
        fprintf(stderr, PROGRAM ": %s, %.2f s at %.0f Hz\n", o.wav_path,
                (double)b.count / vx_sample_rate(chip), vx_sample_rate(chip));

    vx_destroy(chip);
    free(phones);
    free(b.pcm);
    return status;
}
