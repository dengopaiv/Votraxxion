// pybind11 bindings for VotraxSC01ACore
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "votrax_core.h"

namespace py = pybind11;

PYBIND11_MODULE(_votrax_core, m) {
    m.doc() = "Votrax SC-01A C++ DSP core";

    py::class_<VotraxSC01ACore>(m, "VotraxSC01ACore")
        .def(py::init<bool>(), py::arg("enhanced") = false)
        .def("reset", &VotraxSC01ACore::reset,
             "Power-on reset: initialize all state to defaults.")
        .def("phone_commit", &VotraxSC01ACore::phone_commit,
             py::arg("phone"), py::arg("inflection") = 0,
             "Latch a new phoneme and begin generating it.")
        .def("generate_one_sample", &VotraxSC01ACore::generate_one_sample,
             "Generate a single audio sample at 40 kHz.")
        .def("generate_samples", [](VotraxSC01ACore& self, int n) {
                auto result = py::array_t<double>(n);
                auto buf = result.mutable_unchecked<1>();
                for (int i = 0; i < n; i++) {
                    buf(i) = self.generate_one_sample();
                }
                return result;
            },
            py::arg("n"),
            "Generate n audio samples at 40 kHz, returned as a numpy array.")
        .def_property_readonly("phone_done", &VotraxSC01ACore::phone_done,
             "True when the current phoneme has finished.")
        .def_property_readonly("enhanced", &VotraxSC01ACore::enhanced,
             "True when enhanced mode (KLGLOTT88 + PolyBLEP) is active.");

    // Expose clock constants for consistency checks
    m.attr("SCLOCK") = SCLOCK;
    m.attr("CCLOCK") = CCLOCK;
}
