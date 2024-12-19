#pragma once
#include "result.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

inline uint32_t __attribute__((always_inline)) SubstructCoeff(uint32_t rel_co_popcnt,
                                                              uint32_t common_popcnt) noexcept
{
    uint32_t sum = rel_co_popcnt + common_popcnt;
    return sum ? common_popcnt / sum : 0;
}

inline float __attribute__((always_inline)) TanimotoCoeff(uint32_t common_popcnt,
                                                          uint32_t qcount,
                                                          uint32_t ocount) noexcept
{
    return static_cast<float>(common_popcnt) / (qcount + ocount - common_popcnt);
}

inline float __attribute__((always_inline)) TverskyCoeff(uint32_t common_popcnt,
                                                         uint32_t rel_co_popcnt,
                                                         uint32_t rel_co_popcnt2,
                                                         float a,
                                                         float b) noexcept
{
    float denominator = common_popcnt + a * rel_co_popcnt + b * rel_co_popcnt2;
    return denominator > 0.0f ? common_popcnt / denominator : 0.0f;
}

py::array_t<uint32_t> SubstructureScreenout(const py::array_t<uint64_t> py_query,
                                            const py::array_t<uint64_t> py_db,
                                            uint32_t start,
                                            uint32_t end) noexcept;

py::array_t<Result> TanimotoSearch(const py::array_t<uint64_t> py_query,
                                   const py::array_t<uint64_t> py_db,
                                   float threshold,
                                   uint32_t start,
                                   uint32_t end) noexcept;

py::array_t<Result> TverskySearch(const py::array_t<uint64_t> py_query,
                                  const py::array_t<uint64_t> py_db,
                                  float threshold,
                                  float a,
                                  float b,
                                  uint32_t start,
                                  uint32_t end) noexcept;