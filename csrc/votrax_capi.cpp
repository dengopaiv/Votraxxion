// Implementation of the flat C API — see votrax_capi.h.
//
// There is deliberately very little here.  The chip core and the front end are
// header-only C++; this file exists only to give them a C ABI and to own the
// one piece of state the C API adds, the latched inflection level.

#define VOTRAX_BUILD_DLL 1
#define _USE_MATH_DEFINES 1   // M_PI on MSVC; filters.h wants it

#include "votrax_capi.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <new>
#include <string>
#include <vector>

#include "ttv.h"
#include "votrax_core.h"

struct vx_chip {
    VotraxSC01ACore core;
    int inflection = VX_NEUTRAL_INFLECTION;

    // The scheduler.  `queue` is what has been asked for, `hold` is how many
    // samples the phone now sounding still owns.  Rate control is entirely
    // the arithmetic that sets `hold`.
    std::deque<unsigned char> queue;
    double speed = 1.0;
    int hold = 0;

    vx_chip(double clock, MaskRevision mask)
        : core(clock, 150.0 / 4000.0, 1.0, 1.0, 1.0, mask) {}

    // Commit the next queued phone and decide how long to hold it.
    //
    // This is the whole of constant-pitch rate control.  The natural length is
    // exact (16 ticks of 4*duration+1 chip updates, two samples each), so at
    // speed 2.0 the phone is committed at half its natural length and the next
    // one starts there.  The formant interpolators simply carry on toward the
    // new targets from wherever they had reached, which is what keeps fast
    // speech sounding like speech rather than like a chopped-up recording.
    // Nothing here touches the master clock, so the glottal period -- the
    // pitch -- is untouched.
    void commit_next() {
        const unsigned char packed = queue.front();
        queue.pop_front();
        const int phone = ttv::phone_of(packed);
        // The contour is stored absolutely but applied relative to the base,
        // so a caller who moves the base transposes the whole contour instead
        // of flattening it.  Only four levels exist, so the ends clip.
        const int level = std::max(0, std::min(3,
            inflection + ttv::inflection_of(packed) - ttv::NEUTRAL_INFLECTION));
        core.phone_commit(phone, level);
        const double natural = core.phone_samples(phone);
        hold = static_cast<int>(std::lround(natural / speed));
        if (hold < 1) hold = 1;
    }
};

namespace {

MaskRevision to_mask(int mask) {
    return mask == VX_MASK_SC01 ? MaskRevision::SC01 : MaskRevision::SC01A;
}

// Copy phone codes into the caller's buffer, reporting the full length even
// when it does not fit.
int emit(const std::vector<std::uint8_t> &phones,
         unsigned char *out, int capacity) {
    const int n = static_cast<int>(phones.size());
    if (out && capacity > 0) {
        const int copied = std::min(n, capacity);
        std::memcpy(out, phones.data(), static_cast<std::size_t>(copied));
    }
    return n;
}

}  // namespace

extern "C" {

vx_chip *vx_create(int mask, unsigned int clock_hz) {
    const double clock = clock_hz ? static_cast<double>(clock_hz)
                                  : static_cast<double>(VX_BASE_CLOCK);
    return new (std::nothrow) vx_chip(clock, to_mask(mask));
}

void vx_destroy(vx_chip *chip) { delete chip; }

void vx_reset(vx_chip *chip) {
    if (!chip) return;
    chip->core.reset();
    chip->inflection = VX_NEUTRAL_INFLECTION;
    chip->queue.clear();
    chip->hold = 0;
}

void vx_set_clock(vx_chip *chip, unsigned int hz) {
    if (!chip || !hz) return;
    // The core takes its clock at construction, so rebuild in place, keeping
    // the mask.  Callers change rate between utterances, not mid-phone.
    const MaskRevision mask = chip->core.mask();
    chip->core = VotraxSC01ACore(static_cast<double>(hz), 150.0 / 4000.0,
                                 1.0, 1.0, 1.0, mask);
    chip->hold = 0;   // hold times are in samples; the sample rate just moved
}

unsigned int vx_clock(vx_chip *chip) {
    return chip ? static_cast<unsigned int>(std::lround(chip->core.master_clock()))
                : 0u;
}

double vx_sample_rate(vx_chip *chip) {
    return chip ? chip->core.sclock() : 0.0;
}

void vx_set_mask(vx_chip *chip, int mask) {
    if (chip) chip->core.set_mask(to_mask(mask));
}

int vx_mask(vx_chip *chip) {
    if (!chip) return VX_MASK_SC01A;
    return chip->core.mask() == MaskRevision::SC01 ? VX_MASK_SC01 : VX_MASK_SC01A;
}

void vx_inflection(vx_chip *chip, unsigned char level) {
    if (chip) chip->inflection = level & 0x03;
}

int vx_get_inflection(vx_chip *chip) {
    return chip ? chip->inflection : VX_NEUTRAL_INFLECTION;
}

void vx_set_speed(vx_chip *chip, double speed) {
    if (!chip) return;
    chip->speed = std::max(0.1, std::min(10.0, speed));
}

double vx_speed(vx_chip *chip) { return chip ? chip->speed : 1.0; }

int vx_phone_samples(vx_chip *chip, unsigned char phone) {
    return chip ? chip->core.phone_samples(phone & 0x3F) : 0;
}

void vx_write(vx_chip *chip, unsigned char phone) {
    if (!chip) return;
    chip->core.phone_commit(phone & 0x3F, chip->inflection);
    chip->hold = 0;   // hand control back to the scheduler cleanly
}

int vx_speak(vx_chip *chip, const unsigned char *phones, int count) {
    if (!chip || !phones || count <= 0) return chip ? vx_pending(chip) : 0;
    for (int i = 0; i < count; i++) chip->queue.push_back(phones[i]);
    return static_cast<int>(chip->queue.size());
}

int vx_pending(vx_chip *chip) {
    return chip ? static_cast<int>(chip->queue.size()) : 0;
}

void vx_cancel(vx_chip *chip) {
    if (!chip) return;
    chip->queue.clear();
    chip->hold = 0;
    // Reset, then put back the inflection the reset cleared: a cancel must not
    // silently change the voice for whatever is spoken next.
    const int level = chip->inflection;
    chip->core.reset();
    chip->inflection = level;
}

int vx_ready(vx_chip *chip) {
    return chip && chip->core.phone_done() ? 1 : 0;
}

int vx_render(vx_chip *chip, int16_t *buffer, int count) {
    if (!chip || !buffer || count <= 0) return 0;
    for (int i = 0; i < count; i++) {
        // Commit the next phone when the current one's hold expires.  At
        // speed > 1 that happens before the chip would have asked for it,
        // which is the truncation; at speed < 1 it happens after, and the
        // chip sustains the phone in the meantime.
        //
        // Commit before generating and decrement after, so a phone gets
        // exactly `hold` samples.  Decrementing first costs it one.
        if (chip->hold == 0 && !chip->queue.empty()) chip->commit_next();

        double s = chip->core.generate_one_sample();
        s = std::max(-1.0, std::min(1.0, s));
        buffer[i] = static_cast<int16_t>(std::lround(s * 32767.0));

        if (chip->hold > 0) chip->hold--;
    }
    return count;
}

int ttv_translate(const char *text, unsigned char *out, int capacity) {
    if (!text) return 0;
    return emit(ttv::translate(text), out, capacity);
}

int ttv_translate_flat(const char *text, unsigned char *out, int capacity) {
    if (!text) return 0;
    return emit(ttv::translate_flat(text), out, capacity);
}

int ttv_spell(const char *text, unsigned char *out, int capacity) {
    if (!text) return 0;
    return emit(ttv::spell(text), out, capacity);
}

const char *vx_phone_name(int code) {
    return (code >= 0 && code < 64) ? PHONE_NAMES[code] : nullptr;
}

int vx_phone_by_name(const char *name) {
    return name ? ttv::phone_by_name(name) : -1;
}

}  // extern "C"
