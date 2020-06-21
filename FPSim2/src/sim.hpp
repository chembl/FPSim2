#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct Result
{
    uint32_t idx;
    uint32_t mol_id;
    float coeff;
};

__inline float substruct_coeff(uint64_t rel_co_count, uint64_t int_count);

__inline float tanimoto_coeff(uint64_t int_count,
                              uint32_t qcount,
                              uint32_t ocount);

__inline float tversky_coeff(uint64_t int_count,
                             uint64_t rel_co_count,
                             uint64_t rel_co_count2,
                             float a,
                             float b);

uint64_t py_popcount(py::array_t<uint64_t> pyquery);

void sort_results(py::array_t<Result> pyres);

py::array_t<uint32_t> _substructure_search(py::array_t<uint64_t> pyquery,
                                           py::array_t<uint64_t> pydb,
                                           float threshold,
                                           float a,
                                           float b,
                                           uint8_t sim_type,
                                           uint32_t i_start,
                                           uint32_t i_end);

py::array_t<Result> _similarity_search(py::array_t<uint64_t> pyquery,
                                       py::array_t<uint64_t> pydb,
                                       float threshold,
                                       float a,
                                       float b,
                                       uint8_t sim_type,
                                       uint32_t i_start,
                                       uint32_t i_end);