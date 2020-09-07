#include "sim.hpp"
#include "utils.hpp"
#include "result.hpp"
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
           SubstructureScreenout
           SimilaritySearch
           PyPopcount
           BitStrToIntList
           SortResults
    )pbdoc";

    PYBIND11_NUMPY_DTYPE(Result, idx, mol_id, coeff);

    m.def("SubstructureScreenout", &SubstructureScreenout,
        py::call_guard<py::gil_scoped_release>(), R"pbdoc(
        Substructure search

        Runs a Tversky (a=1, b=0) substructure screenout.
    )pbdoc");

    m.def("SimilaritySearch", &SimilaritySearch,
        py::call_guard<py::gil_scoped_release>(), R"pbdoc(
        Similarity search

        Runs a similarity (Tanimoto or Tversky) search.
    )pbdoc");

    auto mutils = m.def_submodule("utils");

    mutils.def("PyPopcount", &utils::PyPopcount, py::call_guard<py::gil_scoped_release>(),
        R"pbdoc(
        Calc popcount
        
        Calcs the popcount of a NumPy int array.
    )pbdoc");

    mutils.def("BitStrToIntList", &utils::BitStrToIntList,
        R"pbdoc(
        Bitstring to Python int list

        Converts RDKit FP bitstring into a Python int list.
    )pbdoc");

    mutils.def("SortResults", &utils::SortResults, py::call_guard<py::gil_scoped_release>(),
        R"pbdoc(
        Sort results
        
        Sort, inplace, the results NumPy array.
    )pbdoc");
}
