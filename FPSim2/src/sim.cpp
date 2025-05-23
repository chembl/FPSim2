#include "sim.hpp"
#include "popcnt.hpp"
#include "utils.hpp"
#include "result.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <queue>

namespace py = pybind11;

inline uint32_t SubstructCoeff(const uint32_t &rel_co_popcnt,
                               const uint32_t &common_popcnt)
{
    uint32_t coeff = 0;
    coeff = rel_co_popcnt + common_popcnt;
    if (coeff != 0)
        coeff = common_popcnt / coeff;
    return coeff;
}

py::array_t<uint32_t> SubstructureScreenout(const py::array_t<uint64_t> py_query,
                                            const py::array_t<uint64_t> py_db,
                                            const uint32_t start,
                                            const uint32_t end)
{

    // Direct access to np arrays without checks
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

inline float TverskyCoeff(const uint32_t &common_popcnt,
                          const uint32_t &rel_co_popcnt,
                          const uint32_t &rel_co_popcnt2,
                          const float &a, const float &b)
{
    float coeff = 0.0;
    coeff = common_popcnt + a * rel_co_popcnt + b * rel_co_popcnt2;
    if (coeff != 0.0)
        coeff = common_popcnt / coeff;
    return coeff;
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
    std::sort(results->begin(), results->end(), utils::ResultComparator());

    // acquire the GIL before calling Python code
    py::gil_scoped_acquire acquire;
    return utils::Vector2NumPy<Result>(results);
}

struct TanimotoCalculator
{
    static inline float calculate(const uint32_t &common_popcnt,
                                  const uint32_t &qcount,
                                  const uint32_t &ocount)
    {
        return (float)common_popcnt / (qcount + ocount - common_popcnt);
    }
};

struct CosineCalculator
{
    static inline float calculate(const uint32_t &common_popcnt,
                                  const uint32_t &qcount,
                                  const uint32_t &ocount)
    {
        return (float)common_popcnt / sqrt(qcount * ocount);
    }
};

struct DiceCalculator
{
    static inline float calculate(const uint32_t &common_popcnt,
                                  const uint32_t &qcount,
                                  const uint32_t &ocount)
    {
        return (2.0f * common_popcnt) / (qcount + ocount);
    }
};

py::array_t<Result> GenericSearch(const py::array_t<uint64_t> py_query,
                                  const py::array_t<uint64_t> py_db,
                                  const float threshold,
                                  const uint32_t k,
                                  const int calc_type,
                                  const uint32_t start,
                                  const uint32_t end)
{
    switch (calc_type)
    {
    case 0:
        return GenericSearchImpl(py_query, py_db, threshold, k, TanimotoCalculator(), start, end);
    case 1:
        return GenericSearchImpl(py_query, py_db, threshold, k, DiceCalculator(), start, end);
    case 2:
        return GenericSearchImpl(py_query, py_db, threshold, k, CosineCalculator(), start, end);
    default:
        throw std::invalid_argument("Invalid calc_type. Must be 0 (Tanimoto), 1 (Dice), or 2 (Cosine)");
    }
}

template <typename Calculator>
py::array_t<Result> GenericSearchImpl(const py::array_t<uint64_t> py_query,
                                      const py::array_t<uint64_t> py_db,
                                      const float threshold,
                                      const uint32_t k,
                                      Calculator calc,
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
    const auto q_popcnt = qptr[popcnt_idx];

    auto results = new std::vector<Result>;

    if (k > 0) // top-k search
    {
        std::priority_queue<Result, std::vector<Result>, utils::ResultComparator> top_k;

        for (uint32_t idx = start; idx < end; ++idx, dbptr += fp_shape)
        {
            uint64_t common_popcnt = 0;
            for (auto j = 1; j < popcnt_idx; j++)
                common_popcnt += popcntll(qptr[j] & dbptr[j]);

            float coeff = calc.calculate(common_popcnt, q_popcnt, dbptr[popcnt_idx]);
            if (coeff < threshold)
                continue;

            if (top_k.size() < k)
            {
                top_k.push({idx, static_cast<uint32_t>(dbptr[0]), coeff});
            }
            else if (coeff > top_k.top().coeff)
            {
                top_k.pop();
                top_k.push({idx, static_cast<uint32_t>(dbptr[0]), coeff});
            }
        }
        results->reserve(top_k.size());
        while (!top_k.empty())
        {
            results->push_back(top_k.top());
            top_k.pop();
        }
        std::reverse(results->begin(), results->end());
    }
    else // normal search
    {
        for (auto i = start; i < end; i++, dbptr += fp_shape)
        {
            uint64_t common_popcnt = 0;
            for (auto j = 1; j < popcnt_idx; j++)
                common_popcnt += popcntll(qptr[j] & dbptr[j]);

            float coeff = calc.calculate(common_popcnt, q_popcnt, dbptr[popcnt_idx]);
            if (coeff < threshold)
                continue;
            results->push_back({i, static_cast<uint32_t>(dbptr[0]), coeff});
        }
        std::sort(results->begin(), results->end(), utils::ResultComparator());
    }
    // acquire the GIL before calling Python code
    py::gil_scoped_acquire acquire;
    return utils::Vector2NumPy<Result>(results);
}
