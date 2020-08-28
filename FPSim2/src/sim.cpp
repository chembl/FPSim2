#include "sim.hpp"
#include "libpopcnt.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

uint64_t PyPopcount(py::array_t<uint64_t> py_query) {
    auto query = py_query.unchecked<1>();
    uint64_t qcount = 0;
    for (ssize_t i = 0; i < query.shape(0); i++)
        qcount += popcnt64(query(i));
    return qcount;
}

__inline float TanimotoCoeff(uint64_t common_popcnt, uint32_t qcount, 
                             uint32_t ocount) {
    float coeff = 0.0;
    coeff = qcount + ocount - common_popcnt;
    if (coeff != 0.0)
        coeff = common_popcnt / coeff;
    return coeff;
}

__inline float TverskyCoeff(uint64_t common_popcnt, uint64_t rel_co_popcnt, 
                            uint64_t rel_co_popcnt2, float a, float b) {
    float coeff = 0.0;
    coeff = common_popcnt + a * rel_co_popcnt + b * rel_co_popcnt2;
    if (coeff != 0.0)
        coeff = common_popcnt / coeff;
    return coeff;
}

__inline float SubstructCoeff(uint64_t rel_co_popcnt, uint64_t common_popcnt) {
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
    auto db = py_db.unchecked<2>();

    uint8_t popcnt_idx = query.shape(0) - 1;

    // initial similarity result array size, allocate memory for results
    uint32_t results_length = 256;
    auto results = new std::vector<uint32_t>(results_length);

    // relative complement and intersection popcounts
    uint64_t common_popcnt = 0;
    uint64_t rel_co_popcnt = 0;

    // number of results during execution
    uint32_t num_results = 0;

    float coeff = 0.0;

    uint32_t i = i_start;
    while (i_end > i) {
        // calc count for intersection and relative complement
        for (size_t j = 1; j < popcnt_idx; j++) {
            common_popcnt += popcnt64(query(j) & db(i, j));
            rel_co_popcnt += popcnt64(query(j) & ~db(i, j));
        }
        // calc optimised tversky with a=1, b=0
        coeff = SubstructCoeff(rel_co_popcnt, common_popcnt);

        if (coeff == threshold) {
            (*results)[num_results] = db(i, 0);
            num_results += 1;
        }
        if (num_results == results_length) {
            results_length *= 1.12;
            results->resize(results_length);
        }
        //  reset values for next fp
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

void SortResults(py::array_t<Result> pyres) {
    auto res = pyres.unchecked<1>();
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
    auto db = py_db.unchecked<2>();

    uint8_t popcnt_idx = query.shape(0) - 1;

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

    uint32_t i = i_start;
    while (i_end > i) {
        switch (sim_type) {
        case 0: // tanimoto
            for (size_t j = 1; j < popcnt_idx; j++) {
                common_popcnt += popcnt64(query(j) & db(i, j));
            }
            coeff = TanimotoCoeff(common_popcnt, query(popcnt_idx), db(i, popcnt_idx));
            break;
        case 1: // tversky
            for (size_t j = 1; j < popcnt_idx; j++) {
                // popcnts of both relative complements and intersection
                common_popcnt += popcnt64(query(j) & db(i, j));
                rel_co_popcnt += popcnt64(query(j) & ~db(i, j));
                rel_co_popcnt2 += popcnt64(db(i, j) & ~query(j));
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
            (*results)[num_results].mol_id = db(i, 0);
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
