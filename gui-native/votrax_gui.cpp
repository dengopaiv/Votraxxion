/*
 * votrax_gui.cpp - Votrax SC-01 desktop GUI, Win32, no Python.
 *
 * The same shape as the SAM and STSPEECH native GUIs, over the C engine
 * in src/. There is nothing to bundle alongside it: both mask ROMs and
 * the English front end are already compiled into the synthesizer, so
 * this program is six .c files and one window, statically linked, and it
 * runs on a bare Windows with nothing installed first.
 *
 * Plain Win32 controls throughout, deliberately. Every control is a
 * standard one with a static label immediately before it in tab order
 * and an & accelerator, which is what screen readers expect; a custom
 * drawn UI would look the same and be unusable. This program is a front
 * end for a screen reader voice, so that is not a detail.
 *
 * What it exposes that the sibling GUIs have no equivalent for: the two
 * mask revisions as a voice choice, and the chip's two different ways of
 * going faster. Moving the master clock is the 1980 hardware's single
 * knob - tempo and pitch rise together. Speed truncates each phone
 * instead and leaves the clock alone, so tempo moves and pitch does not.
 * Both are here, as separate controls, because they are separate things.
 */

#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS

#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <mmsystem.h>
#include <shellapi.h>   /* CommandLineToArgvW; WIN32_LEAN_AND_MEAN drops it */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wctype.h>

extern "C" {
#include "votrax.h"
}

#include "resource.h"

/* /MT pulls in the CRT but not the UI libraries, so name them here rather
 * than spreading the link line across the build script. */
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "comdlg32.lib")
#pragma comment(lib, "shell32.lib")

/* --------------------------------------------------------------------- */
/* Defaults and ranges                                                   */
/* --------------------------------------------------------------------- */

/*
 * One row per voice parameter, so the ranges, the defaults and the tab
 * order live in a single place.
 *
 * The clock is in kHz because the control is a spin box and 720000 is a
 * silly number to arrow through one at a time; the datasheet part runs at
 * 720 kHz. The range is the one the Workbench presets already use, from
 * the 360 kHz "slow robot" to well past the 1.08 MHz "chipmunk".
 *
 * Speed is a percentage of natural phone length, which is vx_set_speed's
 * factor times a hundred. It starts at 10 because vx_set_speed clamps to
 * [0.1, 10] and there is no point offering a number the engine ignores.
 *
 * Inflection is the chip's 2-bit pitch input. Four levels, nothing
 * between them - about 78, 89, 104 and 125 Hz at the datasheet clock. It
 * is a range of four because the silicon has four.
 */
struct ParamSpec {
    const wchar_t *label;
    int lo, hi, def;
    int labelId, editId, spinId;
};

static const ParamSpec PARAMS[] = {
    { L"C&lock (kHz):", 200, 2000, 720, IDC_CLOCKLABEL, IDC_CLOCK, IDC_CLOCKSPIN },
    { L"&Speed (%):",    10, 1000, 100, IDC_SPEEDLABEL, IDC_SPEED, IDC_SPEEDSPIN },
    { L"&Inflection:",    0,    3,   1, IDC_INFLLABEL,  IDC_INFL,  IDC_INFLSPIN  },
};

#define PARAM_COUNT ((int)(sizeof(PARAMS) / sizeof(PARAMS[0])))

enum { P_CLOCK = 0, P_SPEED, P_INFLECTION };

/* The two production masks, in the order vx_create numbers them. */
static const wchar_t *MASK_NAMES[] = {
    L"SC-01-A (1981)",
    L"SC-01 (1980)",
};

#define MASK_COUNT ((int)(sizeof(MASK_NAMES) / sizeof(MASK_NAMES[0])))

/*
 * Voice presets over the four things this chip actually has: which mask,
 * how fast the clock runs, how hard the scheduler truncates, and where
 * the inflection sits.
 *
 * Chipmunk and Slow robot are the clock figures from
 * presets/factory/*.json, so the two front ends agree about what those
 * names mean. Screen reader is what the NVDA add-on does at speed:
 * truncation, not clock, because a chipmunk at 300 words a minute is
 * unusable.
 *
 * Custom is last and is what the combo shows when the controls match no
 * preset. Selecting it deliberately does nothing - there is no "custom"
 * set of numbers to apply.
 */
struct PresetSpec {
    const wchar_t *name;
    int mask, clockKHz, speedPct, inflection;
};

static const PresetSpec PRESETS[] = {
    { L"SC-01-A, datasheet",  VX_MASK_SC01A,  720,  100, 1 },
    { L"SC-01, 1980 mask",    VX_MASK_SC01,   720,  100, 1 },
    { L"Screen reader",       VX_MASK_SC01A,  720,  300, 1 },
    { L"Chipmunk",            VX_MASK_SC01A, 1080,  100, 1 },
    { L"Slow robot",          VX_MASK_SC01A,  360,  100, 0 },
    { L"Arcade cabinet",      VX_MASK_SC01,   950,  100, 2 },
};

#define PRESET_COUNT ((int)(sizeof(PRESETS) / sizeof(PRESETS[0])))
#define PRESET_CUSTOM PRESET_COUNT      /* the index of the "Custom" item */

static const wchar_t *DEFAULT_TEXT = L"Hello, my name is Votrax S C zero one.";
static const wchar_t *WINDOW_TITLE = L"Votrax SC-01 Speech Synthesizer";

/* Room for one utterance of phones. ttv_translate will not produce more
 * than TTV_PHONES_MAX for one call, and hand-typed phoneme strings are
 * grown to fit instead. */
#define PHONE_BUF 4096

/* --------------------------------------------------------------------- */
/* Globals                                                               */
/* --------------------------------------------------------------------- */

static HINSTANCE g_inst;
static HWND g_main, g_textLabel, g_text, g_phonemeMode, g_mask, g_preset;
static HWND g_edit[PARAM_COUNT], g_spin[PARAM_COUNT];
static HWND g_preview, g_convert, g_render;
static HFONT g_font;

/*
 * Set once every control exists. The spin controls fire EN_CHANGE on
 * their buddy edits as they are created, and the handler for that reads
 * every other control; without this it would run against handles that
 * are still NULL.
 */
static int g_ready;

static volatile LONG g_playing;

/* --------------------------------------------------------------------- */
/* Small helpers                                                         */
/* --------------------------------------------------------------------- */

static void ShowError(const wchar_t *msg, const wchar_t *title)
{
    MessageBoxW(g_main, msg, title, MB_OK | MB_ICONERROR);
}

static void ShowWarn(const wchar_t *msg, const wchar_t *title)
{
    MessageBoxW(g_main, msg, title, MB_OK | MB_ICONWARNING);
}

/*
 * The front end speaks bytes, the UI holds UTF-16. ttv_translate folds
 * its input to ASCII anyway, so anything outside it would be dropped
 * there; CP_ACP keeps the mapping predictable for the Latin-1 range.
 */
static char *WideToBytes(const wchar_t *w)
{
    int n = WideCharToMultiByte(CP_ACP, 0, w, -1, NULL, 0, NULL, NULL);
    char *s;

    if (n <= 0) {
        return NULL;
    }
    s = (char *)malloc((size_t)n);
    if (s == NULL) {
        return NULL;
    }
    if (WideCharToMultiByte(CP_ACP, 0, w, -1, s, n, NULL, NULL) <= 0) {
        free(s);
        return NULL;
    }
    return s;
}

/* Window text as a freshly allocated wide string; caller frees. */
static wchar_t *GetText(HWND hwnd)
{
    int n = GetWindowTextLengthW(hwnd);
    wchar_t *buf = (wchar_t *)malloc(((size_t)n + 1) * sizeof(wchar_t));

    if (buf == NULL) {
        return NULL;
    }
    GetWindowTextW(hwnd, buf, n + 1);
    buf[n] = L'\0';
    return buf;
}

static void TrimInPlace(wchar_t *s)
{
    wchar_t *start = s, *end;

    while (*start && iswspace(*start)) {
        start++;
    }
    if (start != s) {
        memmove(s, start, (wcslen(start) + 1) * sizeof(wchar_t));
    }
    end = s + wcslen(s);
    while (end > s && iswspace(end[-1])) {
        *--end = L'\0';
    }
}

/*
 * The value a parameter box currently holds.
 *
 * The up-down is the authority, not the edit beside it: with
 * UDS_SETBUDDYINT the control parses its buddy's text and clamps it to
 * the range, so UDM_GETPOS32 follows typing and the arrow keys alike and
 * can never return something out of range. Reading the edit text instead
 * would mean re-implementing that parse and that clamp.
 */
static int GetSpin(int which)
{
    return (int)SendMessageW(g_spin[which], UDM_GETPOS32, 0, 0);
}

static int IsChecked(HWND cb)
{
    return SendMessageW(cb, BM_GETCHECK, 0, 0) == BST_CHECKED;
}

static int GetCombo(HWND combo)
{
    return (int)SendMessageW(combo, CB_GETCURSEL, 0, 0);
}

/* --------------------------------------------------------------------- */
/* Voice settings                                                        */
/* --------------------------------------------------------------------- */

struct VoiceSettings {
    int          mask;
    unsigned int clockHz;
    double       speed;
    int          inflection;
};

static VoiceSettings VoiceFrom(int mask, int clockKHz, int speedPct,
                               int inflection)
{
    VoiceSettings v;

    v.mask = (mask == VX_MASK_SC01) ? VX_MASK_SC01 : VX_MASK_SC01A;
    v.clockHz = (unsigned int)clockKHz * 1000u;
    v.speed = (double)speedPct / 100.0;
    v.inflection = inflection;
    return v;
}

static VoiceSettings VoiceFromUI(void)
{
    return VoiceFrom(GetCombo(g_mask), GetSpin(P_CLOCK), GetSpin(P_SPEED),
                     GetSpin(P_INFLECTION));
}

/* --------------------------------------------------------------------- */
/* Presets                                                               */
/* --------------------------------------------------------------------- */

static void SyncPresetCombo(void);

/*
 * Put a value into one parameter box.
 *
 * Setting the up-down's position is enough: UDS_SETBUDDYINT makes it
 * write the number into its buddy edit, so the box the user reads and
 * the value GetSpin() reports move together and cannot disagree.
 */
static void SetParam(int which, int value)
{
    SendMessageW(g_spin[which], UDM_SETPOS32, 0, value);
}

static void ApplyPreset(int index)
{
    const PresetSpec *p;

    if (index < 0 || index >= PRESET_COUNT) {
        return;   /* Custom: there is nothing to apply */
    }
    p = &PRESETS[index];

    SendMessageW(g_mask, CB_SETCURSEL, (WPARAM)p->mask, 0);
    SetParam(P_CLOCK, p->clockKHz);
    SetParam(P_SPEED, p->speedPct);
    SetParam(P_INFLECTION, p->inflection);

    /* Each SetParam above rewrites a buddy edit, which raises EN_CHANGE
     * and re-runs SyncPresetCombo. Those intermediate runs see a half
     * applied voice and land on Custom, so the combo is put right once
     * here, after all four values are in. It settles on the preset just
     * applied, which makes this a fixed point rather than a loop -
     * CB_SETCURSEL sends no notification back. */
    SyncPresetCombo();
}

/*
 * Point the combo at whichever preset the controls currently describe,
 * or at Custom when they describe none.
 *
 * Deriving the selection from the values, rather than remembering what
 * was last chosen, is what keeps the combo honest when someone edits a
 * number by hand - and it means ApplyPreset needs no guard flag, because
 * CB_SETCURSEL does not send CBN_SELCHANGE back.
 */
static void SyncPresetCombo(void)
{
    int mask, clockKHz, speedPct, inflection, i;

    if (!g_ready) {
        return;
    }
    mask = GetCombo(g_mask);
    clockKHz = GetSpin(P_CLOCK);
    speedPct = GetSpin(P_SPEED);
    inflection = GetSpin(P_INFLECTION);

    for (i = 0; i < PRESET_COUNT; i++) {
        const PresetSpec *p = &PRESETS[i];
        if (p->mask == mask && p->clockKHz == clockKHz
            && p->speedPct == speedPct && p->inflection == inflection) {
            SendMessageW(g_preset, CB_SETCURSEL, (WPARAM)i, 0);
            return;
        }
    }
    SendMessageW(g_preset, CB_SETCURSEL, (WPARAM)PRESET_CUSTOM, 0);
}

/* --------------------------------------------------------------------- */
/* Phones in and out                                                     */
/* --------------------------------------------------------------------- */

/*
 * Parse a phoneme string into packed phone bytes.
 *
 * The grammar is one token per phone, whitespace or comma separated:
 *
 *     NAME[:LEVEL]        e.g.  H AH1 L OO PA1  or  AY:3 M
 *
 * NAME is a datasheet phone name and LEVEL an inflection of 0-3. Leaving
 * the level off means the neutral one, which is what an unmarked phone
 * gets from ttv_translate, so Convert-then-Preview reproduces exactly
 * what Preview on the text would have said.
 *
 * vx_phone_by_name is case sensitive against the datasheet's uppercase
 * names; tokens are folded up first so that typing "ah1" works.
 *
 * On failure `err` gets a sentence naming the token that was wrong,
 * because "invalid phoneme string" tells someone with sixty phones in
 * the box nothing at all.
 */
static int ParsePhones(const char *s, unsigned char **outPhones, int *outCount,
                       wchar_t *err, int errCap)
{
    unsigned char *phones = NULL;
    int count = 0, cap = 0, index = 0;

    while (*s != '\0') {
        char name[24];
        size_t len = 0;
        int level = VX_NEUTRAL_INFLECTION;
        int code;

        while (*s != '\0' && (isspace((unsigned char)*s) || *s == ',')) {
            s++;
        }
        if (*s == '\0') {
            break;
        }
        index++;

        while (*s != '\0' && !isspace((unsigned char)*s) && *s != ','
               && len + 1 < sizeof(name)) {
            name[len++] = (char)toupper((unsigned char)*s);
            s++;
        }
        name[len] = '\0';

        /* Anything left of the token is an over-long name; take it with
         * the rest so the error quotes what was actually typed. */
        while (*s != '\0' && !isspace((unsigned char)*s) && *s != ',') {
            s++;
            len = sizeof(name);   /* mark as truncated */
        }

        if (len < sizeof(name) && len >= 3 && name[len - 2] == ':') {
            char digit = name[len - 1];
            if (digit >= '0' && digit <= '3') {
                level = digit - '0';
                name[len - 2] = '\0';
            }
        }

        code = (len < sizeof(name)) ? vx_phone_by_name(name) : -1;
        if (code < 0) {
            _snwprintf(err, (size_t)errCap - 1,
                       L"Phone %d is not a phoneme name: \"%hs\".\r\n\r\n"
                       L"Names are the datasheet's, such as EH3, AH1, PA0 or "
                       L"STOP, optionally followed by :0 to :3 for pitch.",
                       index, name);
            err[errCap - 1] = L'\0';
            free(phones);
            return 0;
        }

        if (count == cap) {
            int grow = cap ? cap * 2 : 256;
            unsigned char *bigger =
                (unsigned char *)realloc(phones, (size_t)grow);
            if (bigger == NULL) {
                free(phones);
                return 0;
            }
            phones = bigger;
            cap = grow;
        }
        phones[count++] = VX_PACK(code, level);
    }

    if (count == 0) {
        _snwprintf(err, (size_t)errCap - 1,
                   L"There are no phonemes here to speak.");
        err[errCap - 1] = L'\0';
        free(phones);
        return 0;
    }

    *outPhones = phones;
    *outCount = count;
    return 1;
}

/* Packed phones back to the text ParsePhones accepts. Caller frees. */
static wchar_t *PhonesToText(const unsigned char *phones, int count)
{
    /* "EH3:0 " is the longest a phone can print as. */
    size_t cap = (size_t)count * 8 + 1;
    wchar_t *out = (wchar_t *)malloc(cap * sizeof(wchar_t));
    size_t at = 0;
    int i;

    if (out == NULL) {
        return NULL;
    }
    for (i = 0; i < count; i++) {
        const char *name = vx_phone_name(VX_PHONE(phones[i]));
        int level = VX_INFLECTION(phones[i]);
        int n;

        if (name == NULL) {
            continue;
        }
        n = _snwprintf(out + at, cap - at, L"%hs", name);
        if (n < 0) {
            break;
        }
        at += (size_t)n;
        if (level != VX_NEUTRAL_INFLECTION && at + 2 < cap) {
            out[at++] = L':';
            out[at++] = (wchar_t)(L'0' + level);
        }
        if (i + 1 < count && at + 1 < cap) {
            out[at++] = L' ';
        }
    }
    out[at] = L'\0';
    return out;
}

/* Text to packed phones, through the compiled-in front end. Caller frees. */
static int TextToPhones(const char *text, unsigned char **outPhones,
                        int *outCount)
{
    unsigned char *phones = (unsigned char *)malloc(PHONE_BUF);
    int n;

    if (phones == NULL) {
        return 0;
    }
    n = ttv_translate(text, phones, PHONE_BUF);
    if (n <= 0) {
        free(phones);
        return 0;
    }
    if (n > PHONE_BUF) {
        /* The front end reports what it would have produced, so a long
         * paragraph is one reallocation rather than a truncation. */
        unsigned char *bigger = (unsigned char *)realloc(phones, (size_t)n);
        if (bigger == NULL) {
            free(phones);
            return 0;
        }
        phones = bigger;
        n = ttv_translate(text, phones, n);
        if (n <= 0) {
            free(phones);
            return 0;
        }
    }
    *outPhones = phones;
    *outCount = n;
    return 1;
}

/*
 * Turn whatever is in the box into phones, reporting why not if it will
 * not go. Runs on the UI thread so the message box is immediate and the
 * worker thread is handed something already known to be good.
 */
static int BuildPhones(const wchar_t *wide, int phonemeMode,
                       unsigned char **outPhones, int *outCount,
                       wchar_t *err, int errCap)
{
    char *input = WideToBytes(wide);
    int ok;

    if (input == NULL) {
        _snwprintf(err, (size_t)errCap - 1, L"Out of memory.");
        err[errCap - 1] = L'\0';
        return 0;
    }
    if (phonemeMode) {
        ok = ParsePhones(input, outPhones, outCount, err, errCap);
    } else {
        ok = TextToPhones(input, outPhones, outCount);
        if (!ok) {
            _snwprintf(err, (size_t)errCap - 1,
                       L"The front end produced no phonemes for that text.");
            err[errCap - 1] = L'\0';
        }
    }
    free(input);
    return ok;
}

/* --------------------------------------------------------------------- */
/* Synthesis                                                             */
/* --------------------------------------------------------------------- */

/*
 * Render packed phones to 16-bit mono PCM. Returns a malloc'd buffer,
 * its sample count, and the sample rate the chip produced them at -
 * which is the master clock over 18 and therefore moves with the clock
 * control, so the WAV header has to be written from it rather than from
 * a constant.
 *
 * Two details of the chip's protocol show up here. The queue holds
 * VX_QUEUE_CAPACITY phones and drops the rest, so a long utterance is
 * topped up as it drains rather than pushed in one go. And vx_pending
 * reaching zero does not mean silence: the chip is still voicing the
 * last phone it was given, which is what the quarter-second tail at the
 * end is for. The NVDA driver uses the same figure.
 */
static short *RenderPhones(const unsigned char *phones, int count,
                           const VoiceSettings *v, int *outSamples,
                           int *outRate)
{
    vx_chip *chip = vx_create(v->mask, v->clockHz);
    short *pcm = NULL;
    int cap = 0, n = 0, queued = 0, rate, limit, tail, block;

    if (chip == NULL) {
        return NULL;
    }
    vx_set_speed(chip, v->speed);
    vx_inflection(chip, (unsigned char)v->inflection);

    rate = (int)(vx_sample_rate(chip) + 0.5);
    if (rate <= 0) {
        vx_destroy(chip);
        return NULL;
    }
    block = 1024;
    tail = rate / 4;
    /* Ten minutes. Not a limit anyone reaches by speaking; a stop rather
     * than a hang if a phone ever failed to retire. */
    limit = rate * 600;

    for (;;) {
        int done = (queued >= count) && (vx_pending(chip) == 0);

        if (queued < count) {
            int room = VX_QUEUE_CAPACITY - vx_pending(chip);
            if (room > 0) {
                int take = (count - queued < room) ? count - queued : room;
                vx_speak(chip, phones + queued, take);
                queued += take;
                done = 0;
            }
        }
        if (done || n >= limit) {
            break;
        }
        if (n + block > cap) {
            int grow = cap ? cap * 2 : rate;
            short *bigger;
            while (grow < n + block) {
                grow *= 2;
            }
            bigger = (short *)realloc(pcm, (size_t)grow * sizeof(short));
            if (bigger == NULL) {
                free(pcm);
                vx_destroy(chip);
                return NULL;
            }
            pcm = bigger;
            cap = grow;
        }
        n += vx_render(chip, pcm + n, block);
    }

    if (n + tail > cap) {
        short *bigger = (short *)realloc(pcm, (size_t)(n + tail) * sizeof(short));
        if (bigger == NULL) {
            free(pcm);
            vx_destroy(chip);
            return NULL;
        }
        pcm = bigger;
        cap = n + tail;
    }
    n += vx_render(chip, pcm + n, tail);

    vx_destroy(chip);

    if (pcm == NULL || n <= 0) {
        free(pcm);
        return NULL;
    }
    *outSamples = n;
    *outRate = rate;
    return pcm;
}

/* Wrap PCM in a RIFF/WAVE header. Returns a malloc'd buffer.
 *
 * 16-bit signed mono at whatever rate the chip ran at: 40 kHz at the
 * datasheet clock, 20 kHz at 360 kHz, 60 kHz at 1.08 MHz. */
static unsigned char *MakeWav(const short *pcm, int samples, int rate,
                              DWORD *outLen)
{
    const DWORD dataSize = (DWORD)samples * 2;
    const DWORD total = 44 + dataSize;
    unsigned char *w = (unsigned char *)malloc(total);
    DWORD v;

    if (w == NULL) {
        return NULL;
    }
    memcpy(w, "RIFF", 4);
    v = total - 8;             memcpy(w + 4, &v, 4);
    memcpy(w + 8, "WAVEfmt ", 8);
    v = 16;                    memcpy(w + 16, &v, 4);
    { WORD f = 1;              memcpy(w + 20, &f, 2); }   /* PCM */
    { WORD ch = 1;             memcpy(w + 22, &ch, 2); }
    v = (DWORD)rate;           memcpy(w + 24, &v, 4);
    v = (DWORD)rate * 2;       memcpy(w + 28, &v, 4);     /* byte rate */
    { WORD ba = 2;             memcpy(w + 32, &ba, 2); }
    { WORD bits = 16;          memcpy(w + 34, &bits, 2); }
    memcpy(w + 36, "data", 4);
    memcpy(w + 40, &dataSize, 4);
    memcpy(w + 44, pcm, dataSize);

    *outLen = total;
    return w;
}

static int WriteWholeFile(const wchar_t *path, const void *data, DWORD len)
{
    DWORD written = 0;
    HANDLE f = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, NULL);

    if (f == INVALID_HANDLE_VALUE) {
        return 0;
    }
    if (!WriteFile(f, data, len, &written, NULL) || written != len) {
        CloseHandle(f);
        return 0;
    }
    CloseHandle(f);
    return 1;
}

/* --------------------------------------------------------------------- */
/* Preview, on a worker thread                                           */
/* --------------------------------------------------------------------- */

struct PlayJob {
    unsigned char *phones;
    int            count;
    VoiceSettings  voice;
};

static DWORD WINAPI PlayThread(LPVOID param)
{
    PlayJob *job = (PlayJob *)param;
    int samples = 0, rate = 0;
    short *pcm = RenderPhones(job->phones, job->count, &job->voice,
                              &samples, &rate);

    if (pcm != NULL) {
        DWORD wavLen = 0;
        unsigned char *wav = MakeWav(pcm, samples, rate, &wavLen);
        free(pcm);
        if (wav != NULL) {
            /* Synchronous on this thread, so the buffer outlives playback
             * and a second Preview cannot pull it out from underneath. */
            PlaySoundW((LPCWSTR)wav, NULL, SND_MEMORY | SND_SYNC);
            free(wav);
        }
    } else {
        PostMessageW(g_main, WM_APP_SYNTH_FAILED, 0, 0);
    }

    free(job->phones);
    delete job;
    InterlockedExchange(&g_playing, 0);
    PostMessageW(g_main, WM_APP_PLAY_DONE, 0, 0);
    return 0;
}

static void OnPreview(void)
{
    wchar_t *w = GetText(g_text);
    wchar_t err[512];
    unsigned char *phones = NULL;
    int count = 0;
    PlayJob *job;
    HANDLE th;

    if (w == NULL) {
        return;
    }
    TrimInPlace(w);
    if (w[0] == L'\0') {
        free(w);
        ShowWarn(L"Please enter some text to speak.", L"No Text");
        return;
    }
    err[0] = L'\0';
    if (!BuildPhones(w, IsChecked(g_phonemeMode), &phones, &count,
                     err, (int)(sizeof(err) / sizeof(err[0])))) {
        free(w);
        ShowError(err[0] ? err : L"Failed to translate that.", L"Error");
        return;
    }
    free(w);

    if (InterlockedCompareExchange(&g_playing, 1, 0) != 0) {
        free(phones);
        return;   /* already speaking */
    }

    job = new PlayJob;
    job->phones = phones;
    job->count = count;
    /* Read the controls on the UI thread; the worker must not touch them. */
    job->voice = VoiceFromUI();

    EnableWindow(g_preview, FALSE);
    th = CreateThread(NULL, 0, PlayThread, job, 0, NULL);
    if (th == NULL) {
        EnableWindow(g_preview, TRUE);
        InterlockedExchange(&g_playing, 0);
        free(job->phones);
        delete job;
        return;
    }
    CloseHandle(th);
}

/* --------------------------------------------------------------------- */
/* Convert to phonemes                                                   */
/* --------------------------------------------------------------------- */

static void OnConvert(void)
{
    wchar_t *w = GetText(g_text);
    unsigned char *phones = NULL;
    wchar_t *out;
    char *input;
    int count = 0;

    if (w == NULL) {
        return;
    }
    TrimInPlace(w);
    if (w[0] == L'\0') {
        free(w);
        ShowWarn(L"Please enter some text.", L"No Text");
        return;
    }
    if (IsChecked(g_phonemeMode)) {
        free(w);
        ShowWarn(L"The box already holds phonemes. Clear Phoneme mode to "
                 L"convert text.", L"Already Phonemes");
        return;
    }

    input = WideToBytes(w);
    free(w);
    if (input == NULL) {
        return;
    }
    if (!TextToPhones(input, &phones, &count)) {
        free(input);
        ShowError(L"Conversion failed.", L"Error");
        return;
    }
    free(input);

    out = PhonesToText(phones, count);
    free(phones);
    if (out == NULL) {
        ShowError(L"Out of memory.", L"Error");
        return;
    }

    SetWindowTextW(g_text, out);
    free(out);
    SendMessageW(g_phonemeMode, BM_SETCHECK, BST_CHECKED, 0);
    SetWindowTextW(g_textLabel, L"&Phonemes to speak:");
    SetFocus(g_text);
}

/* --------------------------------------------------------------------- */
/* Render to WAV                                                         */
/* --------------------------------------------------------------------- */

static void OnRender(void)
{
    wchar_t *w = GetText(g_text);
    wchar_t path[MAX_PATH] = L"speech.wav";
    wchar_t err[512];
    OPENFILENAMEW ofn;
    VoiceSettings voice;
    unsigned char *phones = NULL, *wav;
    short *pcm;
    DWORD wavLen = 0;
    int count = 0, samples = 0, rate = 0;

    if (w == NULL) {
        return;
    }
    TrimInPlace(w);
    if (w[0] == L'\0') {
        free(w);
        ShowWarn(L"Please enter some text to speak.", L"No Text");
        return;
    }
    err[0] = L'\0';
    if (!BuildPhones(w, IsChecked(g_phonemeMode), &phones, &count,
                     err, (int)(sizeof(err) / sizeof(err[0])))) {
        free(w);
        ShowError(err[0] ? err : L"Failed to translate that.", L"Error");
        return;
    }
    free(w);

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_main;
    ofn.lpstrFilter = L"WAV files (*.wav)\0*.wav\0All files\0*.*\0";
    ofn.lpstrFile = path;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = L"Save WAV file";
    ofn.lpstrDefExt = L"wav";
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_EXPLORER;

    if (!GetSaveFileNameW(&ofn)) {
        free(phones);
        return;   /* cancelled */
    }

    voice = VoiceFromUI();
    pcm = RenderPhones(phones, count, &voice, &samples, &rate);
    free(phones);
    if (pcm == NULL) {
        ShowError(L"Failed to synthesize audio.", L"Error");
        return;
    }

    wav = MakeWav(pcm, samples, rate, &wavLen);
    free(pcm);
    if (wav == NULL) {
        ShowError(L"Out of memory.", L"Error");
        return;
    }

    if (!WriteWholeFile(path, wav, wavLen)) {
        free(wav);
        ShowError(L"Could not write the file.", L"Error");
        return;
    }
    free(wav);

    {
        wchar_t msg[MAX_PATH + 96];
        _snwprintf(msg, MAX_PATH + 95, L"Saved to %s\r\n\r\n%d Hz, %d samples",
                   path, rate, samples);
        msg[MAX_PATH + 95] = L'\0';
        MessageBoxW(g_main, msg, L"Success", MB_OK | MB_ICONINFORMATION);
    }
}

/* --------------------------------------------------------------------- */
/* Layout                                                                */
/* --------------------------------------------------------------------- */

static HWND Make(const wchar_t *cls, const wchar_t *text, DWORD style,
                 int x, int y, int w, int h, int id)
{
    HWND c = CreateWindowExW(0, cls, text, WS_CHILD | WS_VISIBLE | style,
                             x, y, w, h, g_main, (HMENU)(INT_PTR)id,
                             g_inst, NULL);
    if (c != NULL) {
        SendMessageW(c, WM_SETFONT, (WPARAM)g_font, TRUE);
    }
    return c;
}

/*
 * A multiline EDIT answers WM_GETDLGCODE with DLGC_WANTALLKEYS, meaning
 * "give me every key and do not interpret any of them". IsDialogMessage
 * honours that by returning early - before it reaches its own VK_TAB
 * handling - so Tab is delivered to the edit control, which inserts a
 * tab character. Focus goes into the text box and cannot get out.
 *
 * That is a keyboard trap. In a program whose whole purpose is a screen
 * reader voice it is the worst kind, because the people most likely to
 * meet it are the people least able to reach for the mouse instead.
 *
 * So the control keeps DLGC_WANTALLKEYS for everything except a Tab
 * keydown, where it stands aside and lets the dialog manager move the
 * focus. Enter still inserts a newline (ES_WANTRETURN), the arrow keys
 * still navigate the text, and typing is untouched: the answer is only
 * changed for the one key being asked about, which is why WM_GETDLGCODE
 * carries the message it is asking on behalf of.
 *
 * The tab character is not worth keeping. This box holds a sentence to
 * be spoken aloud, and a tab in it is silent.
 */
static WNDPROC g_textProc;

static LRESULT CALLBACK TextProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    LRESULT code;
    const MSG *m;

    if (msg == WM_GETDLGCODE) {
        code = CallWindowProcW(g_textProc, hwnd, msg, wp, lp);
        m = (const MSG *)lp;
        if (m != NULL && m->message == WM_KEYDOWN && m->wParam == VK_TAB) {
            /* DLGC_WANTMESSAGE is the same bit as DLGC_WANTALLKEYS. */
            code &= ~(LRESULT)(DLGC_WANTALLKEYS | DLGC_WANTTAB);
        }
        return code;
    }
    return CallWindowProcW(g_textProc, hwnd, msg, wp, lp);
}

static HWND MakeSpin(HWND buddy, int lo, int hi, int value, int id)
{
    HWND s = CreateWindowExW(0, UPDOWN_CLASSW, NULL,
                             WS_CHILD | WS_VISIBLE | UDS_SETBUDDYINT
                             | UDS_ALIGNRIGHT | UDS_ARROWKEYS | UDS_NOTHOUSANDS,
                             0, 0, 0, 0, g_main, (HMENU)(INT_PTR)id,
                             g_inst, NULL);
    if (s != NULL) {
        SendMessageW(s, UDM_SETBUDDY, (WPARAM)buddy, 0);
        SendMessageW(s, UDM_SETRANGE32, lo, hi);
        SendMessageW(s, UDM_SETPOS32, 0, value);
    }
    return s;
}

static void CreateControls(void)
{
    const int M = 12;            /* margin */
    const int W = 470;           /* client width used for layout */
    const int LBL = 20, ROW = 26, GAP = 8;
    const int labW = 100, edW = 70;
    int y = M;
    int i;

    g_textLabel = Make(L"STATIC", L"&Text to speak:", 0,
                       M, y, W - 2 * M, LBL, IDC_TEXTLABEL);
    y += LBL + 2;

    g_text = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", DEFAULT_TEXT,
                             WS_CHILD | WS_VISIBLE | WS_TABSTOP | WS_VSCROLL
                             | ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN,
                             M, y, W - 2 * M, 110, g_main,
                             (HMENU)(INT_PTR)IDC_TEXT, g_inst, NULL);
    SendMessageW(g_text, WM_SETFONT, (WPARAM)g_font, TRUE);
    g_textProc = (WNDPROC)(LONG_PTR)SetWindowLongPtrW(
        g_text, GWLP_WNDPROC, (LONG_PTR)TextProc);
    y += 110 + GAP;

    g_phonemeMode = Make(L"BUTTON", L"Phoneme &mode (input is phone names)",
                         BS_AUTOCHECKBOX | WS_TABSTOP,
                         M, y, W - 2 * M, LBL, IDC_PHONEMEMODE);
    y += LBL + GAP;

    /* Each label goes in immediately before the control it names, so it
     * precedes it in z-order - which is both the tab order and the order
     * a screen reader reads. The height given to a combo is its
     * dropped-down height; the closed control sizes itself to the font. */
    Make(L"STATIC", L"Mas&k:", SS_RIGHT, M, y + 4, labW, LBL, IDC_MASKLABEL);
    g_mask = Make(L"COMBOBOX", NULL,
                  WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWNLIST,
                  M + labW + GAP, y, 180, 200, IDC_MASK);
    for (i = 0; i < MASK_COUNT; i++) {
        SendMessageW(g_mask, CB_ADDSTRING, 0, (LPARAM)MASK_NAMES[i]);
    }
    SendMessageW(g_mask, CB_SETCURSEL, VX_MASK_SC01A, 0);
    y += ROW + 4;

    /* "V&oice", not "&Voice": Preview owns Alt+V, and two controls
     * sharing an accelerator makes it cycle between them instead of
     * activating either. */
    Make(L"STATIC", L"V&oice preset:", SS_RIGHT, M, y + 4, labW, LBL,
         IDC_PRESETLABEL);
    g_preset = Make(L"COMBOBOX", NULL,
                    WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWNLIST,
                    M + labW + GAP, y, 180, 220, IDC_PRESET);
    for (i = 0; i < PRESET_COUNT; i++) {
        SendMessageW(g_preset, CB_ADDSTRING, 0, (LPARAM)PRESETS[i].name);
    }
    SendMessageW(g_preset, CB_ADDSTRING, 0, (LPARAM)L"Custom");
    SendMessageW(g_preset, CB_SETCURSEL, 0, 0);
    y += ROW + 4;

    for (i = 0; i < PARAM_COUNT; i++) {
        const ParamSpec *p = &PARAMS[i];
        wchar_t buf[16];

        Make(L"STATIC", p->label, SS_RIGHT, M, y + 4, labW, LBL, p->labelId);

        _snwprintf(buf, 15, L"%d", p->def);
        buf[15] = L'\0';
        g_edit[i] = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", buf,
                                    WS_CHILD | WS_VISIBLE | WS_TABSTOP
                                    | ES_NUMBER, M + labW + GAP, y, edW, 23,
                                    g_main, (HMENU)(INT_PTR)p->editId,
                                    g_inst, NULL);
        SendMessageW(g_edit[i], WM_SETFONT, (WPARAM)g_font, TRUE);
        g_spin[i] = MakeSpin(g_edit[i], p->lo, p->hi, p->def, p->spinId);
        y += ROW + 4;
    }

    y += GAP - 4;

    Make(L"STATIC",
         L"Clock moves tempo and pitch together. Speed moves tempo alone.",
         0, M, y, W - 2 * M, LBL, IDC_HELPCLOCK);
    y += LBL;
    Make(L"STATIC",
         L"Phonemes: H AH1 L OO PA1 - add :0 to :3 for per-phone pitch.",
         0, M, y, W - 2 * M, LBL, IDC_HELPPHONES);
    y += LBL + GAP;

    {
        const int bh = 28, bgap = 8;
        int bx = M;

        g_preview = Make(L"BUTTON", L"Pre&view",
                         BS_DEFPUSHBUTTON | WS_TABSTOP, bx, y, 90, bh,
                         IDC_PREVIEW);
        bx += 90 + bgap;
        g_convert = Make(L"BUTTON", L"&Convert to Phonemes",
                         BS_PUSHBUTTON | WS_TABSTOP, bx, y, 170, bh,
                         IDC_CONVERT);
        bx += 170 + bgap;
        g_render = Make(L"BUTTON", L"Render to &WAV",
                        BS_PUSHBUTTON | WS_TABSTOP, bx, y, 130, bh,
                        IDC_RENDER);
    }

    /* Everything exists now, so the EN_CHANGE handler may safely read it. */
    g_ready = 1;
    SyncPresetCombo();
}

/* --------------------------------------------------------------------- */
/* Window procedure                                                      */
/* --------------------------------------------------------------------- */

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    switch (msg) {
    case WM_CREATE:
        g_main = hwnd;
        CreateControls();
        SetFocus(g_text);
        return 0;

    case WM_COMMAND:
        switch (LOWORD(wp)) {
        case IDC_PREVIEW:
            OnPreview();
            return 0;
        case IDC_CONVERT:
            OnConvert();
            return 0;
        case IDC_RENDER:
            OnRender();
            return 0;
        case IDC_PHONEMEMODE:
            if (HIWORD(wp) == BN_CLICKED) {
                SetWindowTextW(g_textLabel,
                               IsChecked(g_phonemeMode)
                               ? L"&Phonemes to speak:"
                               : L"&Text to speak:");
            }
            return 0;
        case IDC_PRESET:
            if (HIWORD(wp) == CBN_SELCHANGE) {
                ApplyPreset(GetCombo(g_preset));
            }
            return 0;

        /* Changing the mask, typing in a parameter box or nudging its
         * spin can take the voice off a named preset or land it exactly
         * on one. Either way the combo follows the controls. */
        case IDC_MASK:
            if (HIWORD(wp) == CBN_SELCHANGE) {
                SyncPresetCombo();
            }
            return 0;
        case IDC_CLOCK:
        case IDC_SPEED:
        case IDC_INFL:
            if (HIWORD(wp) == EN_CHANGE) {
                SyncPresetCombo();
            }
            return 0;
        case IDCANCEL:
            DestroyWindow(hwnd);
            return 0;
        }
        break;

    case WM_APP_PLAY_DONE:
        EnableWindow(g_preview, TRUE);
        return 0;

    case WM_APP_SYNTH_FAILED:
        ShowError(L"Failed to synthesize audio.", L"Error");
        return 0;

    case WM_CTLCOLORSTATIC:
        /* Let the labels sit on the dialog background rather than white. */
        SetBkMode((HDC)wp, TRANSPARENT);
        return (LRESULT)GetSysColorBrush(COLOR_BTNFACE);

    case WM_CLOSE:
        DestroyWindow(hwnd);
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

/* --------------------------------------------------------------------- */
/* Startup                                                               */
/* --------------------------------------------------------------------- */

static void MakeFont(void)
{
    NONCLIENTMETRICSW ncm;

    ZeroMemory(&ncm, sizeof(ncm));
    ncm.cbSize = sizeof(ncm);
    if (SystemParametersInfoW(SPI_GETNONCLIENTMETRICS, sizeof(ncm), &ncm, 0)) {
        g_font = CreateFontIndirectW(&ncm.lfMessageFont);
    }
    if (g_font == NULL) {
        g_font = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
    }
}

/*
 * Headless self-test, so the shipped binary can be checked rather than a
 * separate harness that merely shares its sources.
 *
 *   votrax_gui.exe --selftest MASK CLOCKKHZ SPEEDPCT INFLECTION
 *                             PHONEMEMODE OUT.WAV TEXT
 *
 * MASK is 0 for the SC-01-A and 1 for the 1980 SC-01, PHONEMEMODE is 0
 * or 1 and means what the checkbox means, and the rest are the numbers
 * in the three parameter boxes. Everything downstream is the code the
 * buttons run - VoiceFrom, BuildPhones, RenderPhones, MakeWav - so a
 * match against a direct drive of the library is a statement about this
 * executable and not about a copy of it. Returns 0 on success. See
 * tools/verify_gui.py.
 *
 * The preset combo needs no argument of its own: choosing a preset only
 * writes the four control values, so passing those values is passing the
 * preset.
 */
static int RunSelfTest(int argc, wchar_t **argv)
{
    VoiceSettings v;
    unsigned char *phones = NULL, *wav;
    short *pcm;
    wchar_t err[512];
    DWORD wavLen = 0;
    int phonemeMode, count = 0, samples = 0, rate = 0;

    if (argc < 9) {
        return 2;
    }
    v = VoiceFrom(_wtoi(argv[2]), _wtoi(argv[3]), _wtoi(argv[4]),
                  _wtoi(argv[5]));
    phonemeMode = _wtoi(argv[6]);

    err[0] = L'\0';
    if (!BuildPhones(argv[8], phonemeMode, &phones, &count,
                     err, (int)(sizeof(err) / sizeof(err[0])))) {
        return 3;
    }

    pcm = RenderPhones(phones, count, &v, &samples, &rate);
    free(phones);
    if (pcm == NULL) {
        return 4;
    }

    wav = MakeWav(pcm, samples, rate, &wavLen);
    free(pcm);
    if (wav == NULL) {
        return 5;
    }
    if (!WriteWholeFile(argv[7], wav, wavLen)) {
        free(wav);
        return 6;
    }
    free(wav);
    return 0;
}

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE, LPWSTR, int show)
{
    WNDCLASSEXW wc;
    INITCOMMONCONTROLSEX icc;
    HWND hwnd;
    MSG msg;
    RECT r;

    g_inst = inst;

    icc.dwSize = sizeof(icc);
    icc.dwICC = ICC_UPDOWN_CLASS | ICC_STANDARD_CLASSES;
    InitCommonControlsEx(&icc);

    {
        int argc = 0;
        wchar_t **argv = CommandLineToArgvW(GetCommandLineW(), &argc);
        if (argv != NULL) {
            if (argc >= 2 && wcscmp(argv[1], L"--selftest") == 0) {
                int rc = RunSelfTest(argc, argv);
                LocalFree(argv);
                return rc;
            }
            LocalFree(argv);
        }
    }

    MakeFont();

    ZeroMemory(&wc, sizeof(wc));
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = WndProc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursorW(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wc.lpszClassName = L"VotraxSC01MainWindow";
    wc.hIcon = LoadIconW(NULL, IDI_APPLICATION);
    wc.hIconSm = LoadIconW(NULL, IDI_APPLICATION);
    if (!RegisterClassExW(&wc)) {
        return 1;
    }

    /* Tall enough for the mask row, the preset row, three parameters and
     * two lines of help: the buttons end at 442. */
    r.left = 0;
    r.top = 0;
    r.right = 470;
    r.bottom = 456;
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME, FALSE);

    hwnd = CreateWindowExW(0, wc.lpszClassName, WINDOW_TITLE,
                           (WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME
                            & ~WS_MAXIMIZEBOX),
                           CW_USEDEFAULT, CW_USEDEFAULT,
                           r.right - r.left, r.bottom - r.top,
                           NULL, NULL, inst, NULL);
    if (hwnd == NULL) {
        return 1;
    }

    ShowWindow(hwnd, show);
    UpdateWindow(hwnd);

    while (GetMessageW(&msg, NULL, 0, 0) > 0) {
        /* IsDialogMessage gives tab order, arrow keys within groups, the
         * & accelerators and the default button - everything a screen
         * reader user needs and none of which a bare loop provides. */
        if (!IsDialogMessageW(hwnd, &msg)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
    return (int)msg.wParam;
}
