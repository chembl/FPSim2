#include "sim.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#ifdef _MSC_VER
#include <nmmintrin.h>
#endif

namespace py = pybind11;

#ifdef _WIN64
inline uint64_t popcntll(uint64_t X)
{
    return _mm_popcnt_u64(X);
}
#else
inline uint64_t popcntll(uint64_t X)
{
    return __builtin_popcountll(X);
}
#endif

uint64_t PyPopcount(py::array_t<uint64_t> py_query) {
    auto query = py_query.unchecked<1>();
    uint64_t qcount = 0;
    for (size_t i = 0; i < query.shape(0); i++)
        qcount += popcntll(query(i));
    return qcount;
}

py::list BitStrToIntList(std::string &bit_string) {
    py::list efp;
    for (size_t i = 0; i < bit_string.length(); i += 64) {
        efp.append(std::stoull(bit_string.substr(i, 64), 0, 2));
    }
    return efp;
}

inline float TanimotoCoeff(uint64_t common_popcnt, uint64_t qcount,
                           uint64_t ocount) {
    float coeff = 0.0;
    coeff = qcount + ocount - common_popcnt;
    if (coeff != 0.0)
        coeff = common_popcnt / coeff;
    return coeff;
}

inline float TverskyCoeff(uint64_t common_popcnt, uint64_t rel_co_popcnt,
                          uint64_t rel_co_popcnt2, float a, float b) {
    float coeff = 0.0;
    coeff = common_popcnt + a * rel_co_popcnt + b * rel_co_popcnt2;
    if (coeff != 0.0)
        coeff = common_popcnt / coeff;
    return coeff;
}

inline float SubstructCoeff(uint64_t rel_co_popcnt, uint64_t common_popcnt) {
    float coeff = 0.0;
    coeff = rel_co_popcnt + common_popcnt;
    if (coeff != 0.0)
        coeff = common_popcnt / coeff;
    return coeff;
}

py::array_t<uint32_t> SubstructureScreenout(py::array_t<uint64_t> py_query,
                                            py::array_t<uint64_t> py_db,
                                            float threshold,
                                            float a,
                                            float b,
                                            uint8_t sim_type,
                                            uint32_t i_start,
                                            uint32_t i_end) {

    // direct access to np arrays without checks
    auto query = py_query.unchecked<1>();
    uint64_t *qptr = (uint64_t *)query.data(0);
    auto db = py_db.unchecked<2>();
    uint64_t *dbptr = (uint64_t *)db.data(0, 0);

    uint8_t qshape = query.shape(0);
    uint8_t popcnt_idx = qshape - 1;

    // initial similarity result array size, allocate memory for results
    uint32_t results_length = 256;
    auto results = new std::vector<uint32_t>(results_length);

    // relative complement and intersection popcounts
    uint64_t common_popcnt = 0;
    uint64_t rel_co_popcnt = 0;

    // number of results during execution
    uint32_t num_results = 0;

    float coeff = 0.0;
    uint64_t fpidx;

    uint32_t i = i_start;
    while (i_end > i) {
        fpidx = i * qshape;
        // calc count for intersection and relative complement
        for (size_t j = 1; j < popcnt_idx; j++) {
            common_popcnt += popcntll(qptr[j] & dbptr[fpidx + j]);
            rel_co_popcnt += popcntll(qptr[j] & ~dbptr[fpidx + j]);
        }
        // calc optimised tversky with a=1, b=0
        coeff = SubstructCoeff(rel_co_popcnt, common_popcnt);

        if (coeff == threshold) {
            (*results)[num_results] = dbptr[fpidx];
            num_results += 1;
        }
        if (num_results == results_length) {
            results_length *= 1.12;
            results->resize(results_length);
        }
        // reset values for next fp
        common_popcnt = 0;
        rel_co_popcnt = 0;
        i++;
    }
    results->resize(num_results);

    // acquire the GIL before calling Python code
    py::gil_scoped_acquire acquire;

    // python object that will free the memory when destroyed
    auto capsule = py::capsule(results, [](void *results) {
        delete reinterpret_cast<std::vector<uint32_t> *>(results);
    });
    return py::array_t<uint32_t>(results->size(), results->data(), capsule);
}

bool cmp(const Result &l, const Result &r) {
    return l.coeff > r.coeff;
}

void SortResults(py::array_t<Result> py_res) {
    auto res = py_res.unchecked<1>();
    Result *ptr = (Result *)res.data(0);
    std::sort(&ptr[0], &ptr[res.shape(0)], cmp);
}

py::array_t<Result> SimilaritySearch(py::array_t<uint64_t> py_query,
                                     py::array_t<uint64_t> py_db,
                                     float threshold,
                                     float a,
                                     float b,
                                     uint8_t sim_type,
                                     uint32_t i_start,
                                     uint32_t i_end) {

    // direct access to np arrays without checks
    auto query = py_query.unchecked<1>();
    uint64_t *qptr = (uint64_t *)query.data(0);
    auto db = py_db.unchecked<2>();
    uint64_t *dbptr = (uint64_t *)db.data(0, 0);

    uint8_t qshape = query.shape(0);
    uint8_t popcnt_idx = qshape - 1;

    // initial similarity result array size, allocate memory for results
    uint32_t results_length = 256;
    auto results = new std::vector<Result>(results_length);

    // relative complements and intersection popcounts
    uint64_t common_popcnt = 0;
    uint64_t rel_co_popcnt = 0;
    uint64_t rel_co_popcnt2 = 0;

    // number of results during execution
    uint32_t num_results = 0;

    float coeff = 0.0;
    uint64_t fpidx;

    uint32_t i = i_start;
    while (i_end > i) {
        fpidx = i * qshape;
        switch (sim_type) {
        case 0: // tanimoto
            for (size_t j = 1; j < popcnt_idx; j++)
                common_popcnt += popcntll(qptr[j] & dbptr[fpidx + j]);
            coeff = TanimotoCoeff(common_popcnt, qptr[popcnt_idx], dbptr[fpidx + popcnt_idx]);
            break;
        case 1: // tversky
            for (size_t j = 1; j < popcnt_idx; j++) {
                // popcnts of both relative complements and intersection
                common_popcnt += popcntll(qptr[j] & dbptr[fpidx + j]);
                rel_co_popcnt += popcntll(qptr[j] & ~dbptr[fpidx + j]);
                rel_co_popcnt2 += popcntll(dbptr[fpidx + j] & ~qptr[j]);
            }
            coeff = TverskyCoeff(common_popcnt, rel_co_popcnt, rel_co_popcnt2, a, b);
            rel_co_popcnt = 0;
            rel_co_popcnt2 = 0;
            break;
        default:
            throw std::invalid_argument("Unknown simialirty type");
        }

        if (coeff >= threshold) {
            (*results)[num_results].idx = i;
            (*results)[num_results].mol_id = dbptr[fpidx];
            (*results)[num_results].coeff = coeff;
            num_results += 1;
        }
        if (num_results == results_length) {
            // reallocate memory
            results_length *= 1.12;
            results->resize(results_length);
        }
        common_popcnt = 0;
        i++;
    }
    // set final size and sort
    results->resize(num_results);
    std::sort(results->begin(), results->end(), cmp);

    // acquire the GIL before calling Python code
    py::gil_scoped_acquire acquire;

    // python object that will free the memory when destroyed
    auto capsule = py::capsule(results, [](void *results) {
        delete reinterpret_cast<std::vector<Result> *>(results);
    });
    return py::array_t<Result>(results->size(), results->data(), capsule);
}
