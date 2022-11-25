#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "result.hpp"

namespace py = pybind11;

namespace utils {

    uint64_t PyPopcount(const py::array_t<uint64_t> py_query);

    py::list BitStrToIntList(const std::string &bit_string);

    bool cmp(const Result &l, const Result &r);
    
    void SortResults(py::array_t<Result> py_res);

    // zero-copy C++ vector to NumPy array
    template<typename T>
    inline py::array_t<T> Vector2NumPy(std::vector<T> *vec) {
        // memory freed when the NumPy object is destroyed
        auto free_when_done = py::capsule(vec, [](void* ptr) {
            delete reinterpret_cast<std::vector<T> *>(ptr);
        });
        return py::array_t<T>(vec->size(), vec->data(), free_when_done);
    }
}