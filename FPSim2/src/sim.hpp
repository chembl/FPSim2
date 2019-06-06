#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct Result
{
    uint32_t mol_id;
    float coeff;
};

float substruct_coeff(uint32_t fp_subs, uint32_t int_count);

float tanimoto_coeff(uint32_t int_count,
                     uint32_t count_query,
                     uint32_t count_other);

uint32_t py_popcount(py::array_t<unsigned long long> query);

py::array_t<uint32_t> _substructure_search(py::array_t<unsigned long long> query,
                                           py::array_t<unsigned long long> db,
                                           float threshold,
                                           uint32_t i_start,
                                           uint32_t i_end);

py::array_t<Result> _similarity_search(py::array_t<unsigned long long> query,
                                       py::array_t<unsigned long long> db,
                                       float threshold,
                                       uint32_t i_start,
                                       uint32_t i_end);