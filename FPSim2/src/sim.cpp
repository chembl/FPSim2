#include "sim.hpp"
#include "popcnt.hpp"
#include "utils.hpp"
#include "result.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

inline float TanimotoCoeff(const uint32_t &common_popcnt,
                           const uint32_t &qcount,
                           const uint32_t &ocount) {
    return (float) common_popcnt / (qcount + ocount - common_popcnt);
}

inline float TverskyCoeff(const uint32_t &common_popcnt,
                          const uint32_t &rel_co_popcnt,
                          const uint32_t &rel_co_popcnt2,
                          const float &a, const float &b) {
    float coeff = 0.0;
    coeff = common_popcnt + a * rel_co_popcnt + b * rel_co_popcnt2;
    if (coeff != 0.0)
        coeff = common_popcnt / coeff;
    return coeff;
}

inline uint32_t SubstructCoeff(const uint32_t &rel_co_popcnt,
                               const uint32_t &common_popcnt) {
    uint32_t coeff = 0;
    coeff = rel_co_popcnt + common_popcnt;
    if (coeff != 0)
        coeff = common_popcnt / coeff;
    return coeff;
}

py::array_t<uint32_t> SubstructureScreenout(const py::array_t<uint64_t> py_query,
                                            const py::array_t<uint64_t> py_db,
                                            const float threshold,
                                            const float a,
                                            const float b,
                                            const uint8_t sim_type,
                                            const uint32_t i_start,
                                            const uint32_t i_end) {

    // direct access to np arrays without checks
    const auto query = py_query.unchecked<1>();
    const auto *qptr = (uint64_t *)query.data(0);
    const auto db = py_db.unchecked<2>();
    const auto *dbptr = (uint64_t *)db.data(0, 0);

    const auto qshape = query.shape(0);
    const auto popcnt_idx = qshape - 1;

    auto results = new std::vector<uint32_t>();

    // relative complement and intersection popcounts
    uint64_t common_popcnt = 0;
    uint64_t rel_co_popcnt = 0;

    uint32_t coeff;
    uint64_t fpidx;
    for (auto i = i_start; i < i_end; i++) {
        fpidx = i * qshape;
        // calc count for intersection and relative complement
        for (size_t j = 1; j < popcnt_idx; j++) {
            common_popcnt += popcntll(qptr[j] & dbptr[fpidx + j]);
            rel_co_popcnt += popcntll(qptr[j] & ~dbptr[fpidx + j]);
        }
        // calc optimised tversky with a=1, b=0
        coeff = SubstructCoeff(rel_co_popcnt, common_popcnt);

        if (coeff == 1)
            results->push_back((uint32_t) dbptr[fpidx]);

        // reset values for next fp
        common_popcnt = 0;
        rel_co_popcnt = 0;
    }
    // acquire the GIL before calling Python code
    py::gil_scoped_acquire acquire;
    return utils::Vector2NumPy<uint32_t>(results);
}

py::array_t<Result> SimilaritySearch(const py::array_t<uint64_t> py_query,
                                     const py::array_t<uint64_t> py_db,
                                     const float threshold,
                                     const float a,
                                     const float b,
                                     const uint8_t sim_type,
                                     const uint32_t i_start,
                                     const uint32_t i_end) {

    // direct access to np arrays without checks
    const auto query = py_query.unchecked<1>();
    const auto *qptr = (uint64_t *)query.data(0);
    const auto db = py_db.unchecked<2>();
    const auto *dbptr = (uint64_t *)db.data(0, 0);

    const auto qshape = query.shape(0);
    const auto popcnt_idx = qshape - 1;

    auto results = new std::vector<Result>();

    // relative complements and intersection popcounts
    uint64_t common_popcnt = 0;
    uint64_t rel_co_popcnt = 0;
    uint64_t rel_co_popcnt2 = 0;

    float coeff;
    uint64_t fpidx;
    for (auto i = i_start; i < i_end; i++) {
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

        if (coeff >= threshold)
            results->push_back({i, (uint32_t) dbptr[fpidx], coeff});
        common_popcnt = 0;
    }
    std::sort(results->begin(), results->end(), utils::cmp);

    // acquire the GIL before calling Python code
    py::gil_scoped_acquire acquire;
    return utils::Vector2NumPy<Result>(results);
}
