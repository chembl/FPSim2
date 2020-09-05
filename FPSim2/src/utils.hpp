#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "result.hpp"

namespace py = pybind11;

namespace utils {

    uint64_t PyPopcount(const py::array_t<uint64_t> py_query);

    py::list BitStrToIntList(const std::string &bit_string);

    bool cmp(const Result &l, const Result &r);
    
    void SortResults(py::array_t<Result> py_res);
}