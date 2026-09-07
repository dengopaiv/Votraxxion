// pybind11 bindings for VotraxSC01ACore
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "votrax_core.h"

namespace py = pybind11;

PYBIND11_MODULE(_votrax_core, m) {
    m.doc() = "Votrax SC-01 / SC-01-A C++ DSP core";

    py::enum_<MaskRevision>(m, "MaskRevision",
        "Which SC-01 mask ROM revision to speak with. SC01 is the 1980 part, "
        "SC01A the later revision. They differ only in the voice amplitude of "
        "twelve open vowels; SC01 runs all of them at full scale and is "
        "audibly louder and more strident.")
        .value("SC01", MaskRevision::SC01)
        .value("SC01A", MaskRevision::SC01A);

    py::class_<PhonemeParams>(m, "PhonemeParams")
        .def(py::init([](int f1, int va, int f2, int fc, int f2q,
                         int f3, int fa, int cld, int vd,
                         int closure, int duration, bool pause) {
            return PhonemeParams{f1, va, f2, fc, f2q, f3, fa,
                                 cld, vd, closure, duration, pause};
        }),
             py::arg("f1"), py::arg("va"), py::arg("f2"), py::arg("fc"),
             py::arg("f2q"), py::arg("f3"), py::arg("fa"), py::arg("cld"),
             py::arg("vd"), py::arg("closure"), py::arg("duration"),
             py::arg("pause"))
        .def_readwrite("f1", &PhonemeParams::f1)
        .def_readwrite("va", &PhonemeParams::va)
        .def_readwrite("f2", &PhonemeParams::f2)
        .def_readwrite("fc", &PhonemeParams::fc)
        .def_readwrite("f2q", &PhonemeParams::f2q)
        .def_readwrite("f3", &PhonemeParams::f3)
        .def_readwrite("fa", &PhonemeParams::fa)
        .def_readwrite("cld", &PhonemeParams::cld)
        .def_readwrite("vd", &PhonemeParams::vd)
        .def_readwrite("closure", &PhonemeParams::closure)
        .def_readwrite("duration", &PhonemeParams::duration)
        .def_readwrite("pause", &PhonemeParams::pause);

    py::class_<VotraxSC01ACore>(m, "VotraxSC01ACore")
        .def(py::init<double, double, double, double, double, MaskRevision>(),
             py::arg("master_clock") = DEFAULT_MASTER_CLOCK,
             py::arg("fx_fudge") = 150.0 / 4000.0,
             py::arg("closure_strength") = 1.0,
             py::arg("articulation_rate") = 1.0,
             py::arg("voice_closure_ratio") = 1.0,
             py::arg("mask") = MaskRevision::SC01A,
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
        .def("reset", &VotraxSC01ACore::reset,
             "Power-on reset: initialize all state to defaults.")
        .def("phone_commit", &VotraxSC01ACore::phone_commit,
             py::arg("phone"), py::arg("inflection") = 0,
             "Latch a new phoneme and begin generating it, using ROM params.")
        .def("phone_commit_override", &VotraxSC01ACore::phone_commit_override,
             py::arg("phone"), py::arg("inflection"), py::arg("params"),
             "Latch a phoneme with explicit PhonemeParams, bypassing the ROM.")
        .def_static("rom_params", &VotraxSC01ACore::rom_params,
             py::arg("phone"), py::arg("mask") = MaskRevision::SC01A,
             "Return the ROM-decoded PhonemeParams for a phoneme code (0-63), "
             "for the given mask revision.")
        .def("generate_one_sample", &VotraxSC01ACore::generate_one_sample,
             "Generate a single audio sample at the chip's current SCLOCK.")
        .def("generate_samples", [](VotraxSC01ACore& self, int n) {
                auto result = py::array_t<double>(n);
                auto buf = result.mutable_unchecked<1>();
                for (int i = 0; i < n; i++) {
                    buf(i) = self.generate_one_sample();
                }
                return result;
            },
            py::arg("n"),
            "Generate n audio samples, returned as a numpy array.")
        .def_property_readonly("phone_done", &VotraxSC01ACore::phone_done,
             "True when the current phoneme has finished.")
        .def("phone_samples", &VotraxSC01ACore::phone_samples,
             py::arg("phone"),
             "Natural length of a phoneme in samples at the current clock: "
             "32 * (4 * duration + 1), exact for all 64 phones. Divide by a "
             "speed factor and commit the next phoneme early to change tempo "
             "without changing pitch.")
        .def_property_readonly("master_clock", &VotraxSC01ACore::master_clock,
             "Master clock frequency in Hz (constructor arg).")
        .def_property_readonly("sclock", &VotraxSC01ACore::sclock,
             "Analog sample rate in Hz (master_clock / 18).")
        .def_property_readonly("cclock", &VotraxSC01ACore::cclock,
             "Chip update rate in Hz (master_clock / 36).")
        .def_property_readonly("fx_fudge", &VotraxSC01ACore::fx_fudge,
             "Final-stage lowpass fudge factor (constructor arg).")
        .def_property_readonly("closure_strength", &VotraxSC01ACore::closure_strength,
             "Closure-attenuation scaling (constructor arg).")
        .def_property("articulation_rate",
             &VotraxSC01ACore::articulation_rate,
             &VotraxSC01ACore::set_articulation_rate,
             "Formant-interpolator speed scale. 1.0 = SC-01 native, higher = "
             "faster transitions. Writable during synthesis.")
        .def_property("voice_closure_ratio",
             &VotraxSC01ACore::voice_closure_ratio,
             &VotraxSC01ACore::set_voice_closure_ratio,
             "How much closure dip applies to the voiced path. 1.0 = SC-01 "
             "native (full mute), 0.0 = voice unaffected. Use small values "
             "(0.1-0.3) to keep voiced stops audible while preserving some "
             "stop character. Writable during synthesis.")
        .def_property("mask",
             &VotraxSC01ACore::mask,
             &VotraxSC01ACore::set_mask,
             "Which mask ROM revision is speaking (MaskRevision.SC01 or "
             ".SC01A). Writable; takes effect from the next phone_commit().");

    // Default clock values (derived from nominal 720 kHz master)
    m.attr("DEFAULT_MASTER_CLOCK") = DEFAULT_MASTER_CLOCK;
    m.attr("SCLOCK") = sclock_from_master(DEFAULT_MASTER_CLOCK);
    m.attr("CCLOCK") = cclock_from_master(DEFAULT_MASTER_CLOCK);
}
