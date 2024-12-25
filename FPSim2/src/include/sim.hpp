#pragma once

#include "result.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

inline uint32_t SubstructCoeff(const uint32_t &rel_co_popcnt,
                               const uint32_t &common_popcnt);

inline float TanimotoCoeff(const uint32_t &common_popcnt,
                           const uint32_t &qcount,
                           const uint32_t &ocount);

inline float TverskyCoeff(const uint32_t &common_popcnt,
                          const uint32_t &rel_co_popcnt,
                          const uint32_t &rel_co_popcnt2,
                          const float &a, const float &b);

py::array_t<uint32_t> SubstructureScreenout(const py::array_t<uint64_t> py_query,
                                            const py::array_t<uint64_t> py_db,
                                            const uint32_t start,
                                            const uint32_t end);

py::array_t<Result> TanimotoSearch(const py::array_t<uint64_t> py_query,
                                   const py::array_t<uint64_t> py_db,
                                   const float threshold,
                                   const uint32_t start,
                                   const uint32_t end);

py::array_t<Result> TverskySearch(const py::array_t<uint64_t> py_query,
                                  const py::array_t<uint64_t> py_db,
                                  const float threshold,
                                  const float a,
                                  const float b,
                                  const uint32_t start,
                                  const uint32_t end);

py::array_t<Result> TanimotoSearchTopK(const py::array_t<uint64_t>& py_query,
                                       const py::array_t<uint64_t>& py_db,
                                       const uint32_t k,
                                       const float threshold,
                                       const uint32_t start,
                                       const uint32_t end);
