#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct Result {
    uint32_t idx;
    uint32_t mol_id;
    float coeff;
};

inline float SubstructCoeff(uint64_t rel_co_popcnt, uint64_t common_popcnt);

inline float TanimotoCoeff(uint64_t common_popcnt, uint64_t qcount,
                           uint64_t ocount);

inline float TverskyCoeff(uint64_t common_popcnt, uint64_t rel_co_popcnt,
                          uint64_t rel_co_popcnt2, float a, float b);

uint64_t PyPopcount(py::array_t<uint64_t> pyquery);

void SortResults(py::array_t<Result> pyres);

py::array_t<uint32_t> SubstructureScreenout(py::array_t<uint64_t> py_query,
                                            py::array_t<uint64_t> py_db,
                                            float threshold, float a, float b,
                                            uint8_t sim_type, uint32_t i_start,
                                            uint32_t i_end);

py::array_t<Result> SimilaritySearch(py::array_t<uint64_t> py_query,
                                     py::array_t<uint64_t> py_db,
                                     float threshold, float a, float b,
                                     uint8_t sim_type, uint32_t i_start,
                                     uint32_t i_end);