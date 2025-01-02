#pragma once

#include "result.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array_t<uint32_t> SubstructureScreenout(const py::array_t<uint64_t> py_query,
                                            const py::array_t<uint64_t> py_db,
                                            const uint32_t start,
                                            const uint32_t end);

py::array_t<Result> TverskySearch(const py::array_t<uint64_t> py_query,
                                  const py::array_t<uint64_t> py_db,
                                  const float threshold,
                                  const float a,
                                  const float b,
                                  const uint32_t start,
                                  const uint32_t end);

py::array_t<Result> GenericSearch(const py::array_t<uint64_t> py_query,
                                  const py::array_t<uint64_t> py_db,
                                  const float threshold,
                                  const uint32_t k,
                                  const int calc_type,
                                  const uint32_t start,
                                  const uint32_t end);

template <typename Calculator>
py::array_t<Result> GenericSearchImpl(const py::array_t<uint64_t> py_query,
                                      const py::array_t<uint64_t> py_db,
                                      const float threshold,
                                      const uint32_t k,
                                      Calculator calc,
                                      const uint32_t start,
                                      const uint32_t end);