#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "sim.hpp"

namespace py = pybind11;

PYBIND11_MODULE(FPSim2lib, m)
{
    PYBIND11_NUMPY_DTYPE(Result, mol_id, coeff);
    m.def("_substructure_search", &_substructure_search, py::return_value_policy::take_ownership);
    m.def("_similarity_search", &_similarity_search, py::return_value_policy::take_ownership);
    m.def("py_popcount", &py_popcount), py::call_guard<py::gil_scoped_release>();
}