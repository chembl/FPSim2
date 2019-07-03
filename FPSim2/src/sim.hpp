#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct Result
{
    uint32_t mol_id;
    float coeff;
};

__inline float substruct_coeff(uint32_t rel_co_count, uint32_t int_count);

__inline float tanimoto_coeff(uint32_t int_count,
                              uint32_t qcount,
                              uint32_t ocount);

uint32_t py_popcount(py::array_t<unsigned long long> pyquery);

void sort_results(py::array_t<Result> pyres);

py::array_t<uint32_t> _substructure_search(py::array_t<unsigned long long> pyquery,
                                           py::array_t<unsigned long long> pydb,
                                           float threshold,
                                           uint32_t i_start,
                                           uint32_t i_end);

py::array_t<Result> _similarity_search(py::array_t<unsigned long long> pyquery,
                                       py::array_t<unsigned long long> pydb,
                                       float threshold,
                                       uint32_t i_start,
                                       uint32_t i_end);