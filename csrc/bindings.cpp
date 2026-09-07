// pybind11 bindings for the C chip core.
//
// This is the only C++ left in the project.  The synthesizer under it is C
// (see src/), and this file exists solely to give the Workbench GUI and the
// tests a Python object to hold: `pyvotrax._votrax_core`.
//
// The Python surface is unchanged from when the core itself was C++ -- same
// class, same methods, same properties -- because pyvotrax and 500 tests are
// written against it.  Where the C differs, the difference is absorbed here:
// `pause` is an int in the struct and a bool in Python, and the class is a
// thin holder around a vx_core rather than being the core.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

extern "C" {
#include "votrax_core.h"
#include "votrax_filters.h"
#include "votrax_rom.h"
}

namespace py = pybind11;

namespace {

// A vx_core with a constructor, so pybind11 can own one by value.
struct Core {
    vx_core c;

    Core(double master_clock, double fx_fudge, double closure_strength,
         double articulation_rate, double voice_closure_ratio,
         vx_mask_revision mask) {
        vx_core_init(&c, master_clock, fx_fudge, closure_strength,
                     articulation_rate, voice_closure_ratio, mask);
    }
};

}  // namespace

PYBIND11_MODULE(_votrax_core, m) {
    m.doc() = "Votrax SC-01 / SC-01-A DSP core (C implementation)";

    py::enum_<vx_mask_revision>(m, "MaskRevision",
        "Which SC-01 mask ROM revision to speak with. SC01 is the 1980 part, "
        "SC01A the later revision. They differ only in the voice amplitude of "
        "twelve open vowels; SC01 runs all of them at full scale and is "
        "audibly louder and more strident.")
        .value("SC01", VX_ROM_SC01)
        .value("SC01A", VX_ROM_SC01A);

    py::class_<vx_phoneme>(m, "PhonemeParams")
        .def(py::init([](int f1, int va, int f2, int fc, int f2q,
                         int f3, int fa, int cld, int vd,
                         int closure, int duration, bool pause) {
            vx_phoneme p;
            p.f1 = f1; p.va = va; p.f2 = f2; p.fc = fc; p.f2q = f2q;
            p.f3 = f3; p.fa = fa; p.cld = cld; p.vd = vd;
            p.closure = closure; p.duration = duration; p.pause = pause ? 1 : 0;
            return p;
        }),
             py::arg("f1"), py::arg("va"), py::arg("f2"), py::arg("fc"),
             py::arg("f2q"), py::arg("f3"), py::arg("fa"), py::arg("cld"),
             py::arg("vd"), py::arg("closure"), py::arg("duration"),
             py::arg("pause"))
        .def_readwrite("f1", &vx_phoneme::f1)
        .def_readwrite("va", &vx_phoneme::va)
        .def_readwrite("f2", &vx_phoneme::f2)
        .def_readwrite("fc", &vx_phoneme::fc)
        .def_readwrite("f2q", &vx_phoneme::f2q)
        .def_readwrite("f3", &vx_phoneme::f3)
        .def_readwrite("fa", &vx_phoneme::fa)
        .def_readwrite("cld", &vx_phoneme::cld)
        .def_readwrite("vd", &vx_phoneme::vd)
        .def_readwrite("closure", &vx_phoneme::closure)
        .def_readwrite("duration", &vx_phoneme::duration)
        // A bool in Python, an int in the struct: the tests assert
        // `isinstance(p["pause"], bool)`, and C has no bool in a plain struct
        // we want to keep ABI-simple.
        .def_property("pause",
             [](const vx_phoneme &p) { return p.pause != 0; },
             [](vx_phoneme &p, bool v) { p.pause = v ? 1 : 0; });

    py::class_<Core>(m, "VotraxSC01ACore")
        .def(py::init<double, double, double, double, double, vx_mask_revision>(),
             py::arg("master_clock") = VX_DEFAULT_MASTER_CLOCK,
             py::arg("fx_fudge") = 150.0 / 4000.0,
             py::arg("closure_strength") = 1.0,
             py::arg("articulation_rate") = 1.0,
             py::arg("voice_closure_ratio") = 1.0,
             py::arg("mask") = VX_ROM_SC01A,
             "Construct a chip core. master_clock is in Hz (nominal 720 000, "
             "datasheet-variable for sound-design effects). fx_fudge scales the "
             "final-stage lowpass cutoff: 150/4000 matches MAME's observed "
             "behavior (authentic), 1.0 gives 'as-schematic' 150 Hz. "
             "closure_strength scales plosive closure attenuation "
             "(0.0 = disabled, 1.0 = MAME, >1.0 = exaggerated). "
             "articulation_rate scales the formant-interpolator decay speed "
             "(1.0 = SC-01 native; higher = faster transitions, emulating "
             "SSI-263's programmable articulation register). "
             "voice_closure_ratio (0..1) controls how much closure dip "
             "applies to the voiced path: 1.0 = SC-01 native (voiced "
             "stops go silent during closure), 0.0 = voice path unaffected "
             "(lets /b/ /d/ /g/ keep their buzz through closure). "
             "mask selects the silicon revision's phoneme ROM: SC01A (later, "
             "default) or SC01 (the 1980 part, louder open vowels).")
        .def("reset", [](Core &s) { vx_core_reset(&s.c); },
             "Power-on reset: initialize all state to defaults.")
        .def("phone_commit",
             [](Core &s, int phone, int inflection) {
                 vx_core_phone_commit(&s.c, phone, inflection);
             },
             py::arg("phone"), py::arg("inflection") = 0,
             "Latch a new phoneme and begin generating it, using ROM params.")
        .def("phone_commit_override",
             [](Core &s, int phone, int inflection, const vx_phoneme &params) {
                 vx_core_phone_commit_override(&s.c, phone, inflection, &params);
             },
             py::arg("phone"), py::arg("inflection"), py::arg("params"),
             "Latch a phoneme with explicit PhonemeParams, bypassing the ROM.")
        .def_static("rom_params",
             [](int phone, vx_mask_revision mask) {
                 return vx_rom_phoneme(phone, mask);
             },
             py::arg("phone"), py::arg("mask") = VX_ROM_SC01A,
             "Return the ROM-decoded PhonemeParams for a phoneme code (0-63), "
             "for the given mask revision.")
        .def("generate_one_sample",
             [](Core &s) { return vx_core_generate_one_sample(&s.c); },
             "Generate a single audio sample at the chip's current SCLOCK.")
        .def("generate_samples", [](Core &s, int n) {
                auto result = py::array_t<double>(n);
                auto buf = result.mutable_unchecked<1>();
                for (int i = 0; i < n; i++)
                    buf(i) = vx_core_generate_one_sample(&s.c);
                return result;
            },
            py::arg("n"),
            "Generate n audio samples, returned as a numpy array.")
        .def_property_readonly("phone_done",
             [](const Core &s) { return vx_core_phone_done(&s.c) != 0; },
             "True when the current phoneme has finished.")
        .def("phone_samples",
             [](const Core &s, int phone) {
                 return vx_core_phone_samples(&s.c, phone);
             },
             py::arg("phone"),
             "Natural length of a phoneme in samples at the current clock: "
             "32 * (4 * duration + 1), exact for all 64 phones. Divide by a "
             "speed factor and commit the next phoneme early to change tempo "
             "without changing pitch.")
        .def_property_readonly("master_clock",
             [](const Core &s) { return s.c.master_clock; },
             "Master clock frequency in Hz (constructor arg).")
        .def_property_readonly("sclock",
             [](const Core &s) { return s.c.sclock; },
             "Analog sample rate in Hz (master_clock / 18).")
        .def_property_readonly("cclock",
             [](const Core &s) { return s.c.cclock; },
             "Chip update rate in Hz (master_clock / 36).")
        .def_property_readonly("fx_fudge",
             [](const Core &s) { return s.c.fx_fudge; },
             "Final-stage lowpass fudge factor (constructor arg).")
        .def_property_readonly("closure_strength",
             [](const Core &s) { return s.c.closure_strength; },
             "Closure-attenuation scaling (constructor arg).")
        .def_property("articulation_rate",
             [](const Core &s) { return s.c.articulation_rate; },
             [](Core &s, double v) { s.c.articulation_rate = v; },
             "Formant-interpolator speed scale. 1.0 = SC-01 native, higher = "
             "faster transitions. Writable during synthesis.")
        .def_property("voice_closure_ratio",
             [](const Core &s) { return s.c.voice_closure_ratio; },
             [](Core &s, double v) { s.c.voice_closure_ratio = v; },
             "How much closure dip applies to the voiced path. 1.0 = SC-01 "
             "native (full mute), 0.0 = voice unaffected. Use small values "
             "(0.1-0.3) to keep voiced stops audible while preserving some "
             "stop character. Writable during synthesis.")
        .def_property("mask",
             [](const Core &s) { return s.c.mask; },
             [](Core &s, vx_mask_revision m) { vx_core_set_mask(&s.c, m); },
             "Which mask ROM revision is speaking (MaskRevision.SC01 or "
             ".SC01A). Writable; takes effect from the next phone_commit().");

    // Default clock values (derived from the nominal 720 kHz master).
    m.attr("DEFAULT_MASTER_CLOCK") = VX_DEFAULT_MASTER_CLOCK;
    m.attr("SCLOCK") = vx_sclock_from_master(VX_DEFAULT_MASTER_CLOCK);
    m.attr("CCLOCK") = vx_cclock_from_master(VX_DEFAULT_MASTER_CLOCK);
}
