#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "sim.hpp"

namespace py = pybind11;

PYBIND11_MODULE(FPSim2lib, m)
{
    m.doc() = R"pbdoc(
        FPSim2lib
        ---------
        .. currentmodule:: fpsim2
        .. autosummary::
           :toctree: _generate
           py_popcount
           sort_results
           _substructure_search
           _similarity_search
    )pbdoc";

    PYBIND11_NUMPY_DTYPE(Result, mol_id, coeff);
    
    m.def("py_popcount", &py_popcount, py::call_guard<py::gil_scoped_release>(), R"pbdoc(
        Calc popcount
        Calcs the popcount of a NumPy int array.
    )pbdoc");

    m.def("sort_results", &sort_results, py::call_guard<py::gil_scoped_release>(), R"pbdoc(
        Sort results
        Sorts the NumPy array containing the results.
    )pbdoc");

    m.def("_substructure_search", &_substructure_search, py::call_guard<py::gil_scoped_release>(), R"pbdoc(
        Substructure search
        Runs a substructure search.
    )pbdoc");

    m.def("_similarity_search", &_similarity_search, py::call_guard<py::gil_scoped_release>(), R"pbdoc(
        Similarity search
        Runs a similarity search.
    )pbdoc");
}