#include "sim.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(FPSim2lib, m) {
    m.doc() = R"pbdoc(
        FPSim2lib
        ---------
        .. currentmodule:: FPSim2
        .. autosummary::
           :toctree: _generate
           PyPopcount
           SortResults
           SubstructureScreenout
           SimilaritySearch
    )pbdoc";

    PYBIND11_NUMPY_DTYPE(Result, idx, mol_id, coeff);

    m.def("PyPopcount", &PyPopcount, py::call_guard<py::gil_scoped_release>(),
        R"pbdoc(
        Calc popcount
        
        Calcs the popcount of a NumPy int array.
    )pbdoc");

    m.def("SortResults", &SortResults, py::call_guard<py::gil_scoped_release>(),
        R"pbdoc(
        Sort results
        
        Sort, inplace, the results Numpy array.
    )pbdoc");

    m.def("SubstructureScreenout", &SubstructureScreenout,
        py::call_guard<py::gil_scoped_release>(), R"pbdoc(
        Substructure search

        Runs a substructure search.
    )pbdoc");

    m.def("SimilaritySearch", &SimilaritySearch,
        py::call_guard<py::gil_scoped_release>(), R"pbdoc(
        Similarity search

        Runs a similarity search.
    )pbdoc");
}