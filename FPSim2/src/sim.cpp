#include "sim.hpp"
#include "popcnt.hpp"
#include "utils.hpp"
#include "result.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

py::array_t<uint32_t> SubstructureScreenout(const py::array_t<uint64_t> py_query,
                                            const py::array_t<uint64_t> py_db,
                                            uint32_t start,
                                            uint32_t end)
{

    // direct access to np arrays without checks
    const auto query = py_query.unchecked<1>();
    const auto *qptr = (uint64_t *)query.data(0);
    const auto db = py_db.unchecked<2>();
    const auto *dbptr = (uint64_t *)db.data(start, 0);

    const auto fp_shape = query.shape(0);
    const auto popcnt_idx = fp_shape - 1;

    auto results = new std::vector<uint32_t>();

    uint32_t coeff;
    uint64_t common_popcnt = 0;
    uint64_t rel_co_popcnt = 0;
    for (auto i = start; i < end; i++, dbptr += fp_shape,
              common_popcnt = 0, rel_co_popcnt = 0)
    {
        for (size_t j = 1; j < popcnt_idx; j++)
        {
            common_popcnt += popcntll(qptr[j] & dbptr[j]);
            rel_co_popcnt += popcntll(qptr[j] & ~dbptr[j]);
        }
        // calc optimised tversky with a=1, b=0
        coeff = SubstructCoeff(rel_co_popcnt, common_popcnt);

        if (coeff == 1)
            results->push_back((uint32_t)dbptr[0]);
    }
    // acquire the GIL before calling Python code
    py::gil_scoped_acquire acquire;
    return utils::Vector2NumPy<uint32_t>(results);
}

py::array_t<Result> TanimotoSearch(const py::array_t<uint64_t> py_query,
                                   const py::array_t<uint64_t> py_db,
                                   const float threshold,
                                   const uint32_t start,
                                   const uint32_t end)
{
    // direct access to np arrays without checks
    const auto query = py_query.unchecked<1>();
    const auto *qptr = (uint64_t *)query.data(0);
    const auto db = py_db.unchecked<2>();
    const auto *dbptr = (uint64_t *)db.data(start, 0);

    const auto fp_shape = query.shape(0);
    const auto popcnt_idx = fp_shape - 1;

    auto results = new std::vector<Result>();
    results->reserve((end - start) / 8);

    uint64_t common_popcnt = 0;
    const uint64_t qcount = qptr[popcnt_idx];

    for (auto i = start; i < end; i++, dbptr += fp_shape)
    {
        common_popcnt = 0;
        for (size_t j = 1; j < popcnt_idx; j++)
        {
            common_popcnt += popcntll(qptr[j] & dbptr[j]);
        }

        const float coeff = TanimotoCoeff(common_popcnt, qcount, dbptr[popcnt_idx]);
        if (coeff >= threshold)
        {
            results->push_back({i, static_cast<uint32_t>(dbptr[0]), coeff});
        }
    }
    std::sort(results->begin(), results->end(), utils::cmp);

    py::gil_scoped_acquire acquire;
    return utils::Vector2NumPy<Result>(results);
}

py::array_t<Result> TverskySearch(const py::array_t<uint64_t> py_query,
                                  const py::array_t<uint64_t> py_db,
                                  const float threshold,
                                  const float a,
                                  const float b,
                                  const uint32_t start,
                                  const uint32_t end)
{

    // direct access to np arrays without checks
    const auto query = py_query.unchecked<1>();
    const auto *qptr = (uint64_t *)query.data(0);
    const auto db = py_db.unchecked<2>();
    const auto *dbptr = (uint64_t *)db.data(start, 0);

    const auto fp_shape = query.shape(0);
    const auto popcnt_idx = fp_shape - 1;

    auto results = new std::vector<Result>();

    float coeff;
    uint64_t common_popcnt = 0;
    uint64_t rel_co_popcnt = 0;
    uint64_t rel_co_popcnt2 = 0;
    for (auto i = start; i < end; i++, dbptr += fp_shape,
              common_popcnt = 0, rel_co_popcnt = 0, rel_co_popcnt2 = 0)
    {
        for (auto j = 1; j < popcnt_idx; j++)
        {
            // popcnts of both relative complements and intersection
            common_popcnt += popcntll(qptr[j] & dbptr[j]);
            rel_co_popcnt += popcntll(qptr[j] & ~dbptr[j]);
            rel_co_popcnt2 += popcntll(dbptr[j] & ~qptr[j]);
        }
        coeff = TverskyCoeff(common_popcnt, rel_co_popcnt, rel_co_popcnt2, a, b);
        if (coeff >= threshold)
            results->push_back({i, (uint32_t)dbptr[0], coeff});
    }
    std::sort(results->begin(), results->end(), utils::cmp);

    // acquire the GIL before calling Python code
    py::gil_scoped_acquire acquire;
    return utils::Vector2NumPy<Result>(results);
}