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
                                            uint32_t end) noexcept
{
    const auto query = py_query.unchecked<1>();
    const uint64_t *qptr = query.data(0);
    const auto db = py_db.unchecked<2>();
    const uint64_t *dbptr = db.data(start, 0);

    const size_t fp_shape = query.shape(0);
    const size_t popcnt_idx = fp_shape - 1;

    auto results = new std::vector<uint32_t>();
    results->reserve((end - start) / 8);

    const uint64_t *db_row = dbptr;
    for (uint32_t i = start; i < end; i++)
    {
        uint64_t common_popcnt = 0;
        uint64_t rel_co_popcnt = 0;

        for (size_t j = 1; j < popcnt_idx; j++)
        {
            uint64_t db_bits = db_row[j];
            uint64_t query_bits = qptr[j];
            uint64_t intersection = query_bits & db_bits;
            common_popcnt += popcntll(intersection);
            rel_co_popcnt += popcntll(query_bits & ~db_bits);
        }

        if (SubstructCoeff(rel_co_popcnt, common_popcnt) == 1)
        {
            results->push_back(static_cast<uint32_t>(db_row[0]));
        }

        db_row += fp_shape;
    }

    py::gil_scoped_acquire acquire;
    return utils::Vector2NumPy<uint32_t>(results);
}

py::array_t<Result> TanimotoSearch(const py::array_t<uint64_t> py_query,
                                   const py::array_t<uint64_t> py_db,
                                   float threshold,
                                   uint32_t start,
                                   uint32_t end) noexcept
{
    const uint64_t *qptr = py_query.unchecked<1>().data(0);
    const uint64_t *dbptr = py_db.unchecked<2>().data(start, 0);
    const size_t fp_shape = py_query.shape(0);
    const size_t popcnt_idx = fp_shape - 1;

    auto results = new std::vector<Result>();
    results->reserve((end - start) / 8);

    const uint64_t qcount = qptr[popcnt_idx];
    const uint64_t *db_row = dbptr;

    for (uint32_t i = start; i < end; i++)
    {
        uint64_t common_popcnt = 0;

        for (size_t k = 1; k < popcnt_idx; k++)
        {
            common_popcnt += popcntll(qptr[k] & db_row[k]);
        }

        const float coeff = TanimotoCoeff(common_popcnt, qcount, dbptr[popcnt_idx]);
        if (coeff >= threshold)
        {
            results->push_back({i, static_cast<uint32_t>(db_row[0]), coeff});
        }
        db_row += fp_shape;
    }
    std::sort(results->begin(), results->end(), utils::cmp);

    py::gil_scoped_acquire acquire;
    return utils::Vector2NumPy<Result>(results);
}

py::array_t<Result> TverskySearch(const py::array_t<uint64_t> py_query,
                                  const py::array_t<uint64_t> py_db,
                                  float threshold,
                                  float a,
                                  float b,
                                  uint32_t start,
                                  uint32_t end) noexcept
{
    const auto query = py_query.unchecked<1>();
    const uint64_t *qptr = query.data(0);
    const auto db = py_db.unchecked<2>();
    const uint64_t *dbptr = db.data(start, 0);

    const size_t fp_shape = query.shape(0);
    const size_t popcnt_idx = fp_shape - 1;

    auto results = new std::vector<Result>();
    results->reserve((end - start) / 8);

    const uint64_t *db_row = dbptr;
    for (uint32_t i = start; i < end; i++)
    {
        uint64_t common_popcnt = 0;
        uint64_t rel_co_popcnt = 0;
        uint64_t rel_co_popcnt2 = 0;

        for (size_t j = 1; j < popcnt_idx; j++)
        {
            uint64_t db_bits = db_row[j];
            uint64_t query_bits = qptr[j];
            uint64_t intersection = query_bits & db_bits;
            common_popcnt += popcntll(intersection);
            rel_co_popcnt += popcntll(query_bits & ~db_bits);
            rel_co_popcnt2 += popcntll(db_bits & ~query_bits);
        }

        float coeff = TverskyCoeff(common_popcnt, rel_co_popcnt, rel_co_popcnt2, a, b);
        if (coeff >= threshold)
        {
            results->push_back({i, static_cast<uint32_t>(db_row[0]), coeff});
        }

        db_row += fp_shape;
    }

    std::sort(results->begin(), results->end(), utils::cmp);

    py::gil_scoped_acquire acquire;
    return utils::Vector2NumPy<Result>(results);
}