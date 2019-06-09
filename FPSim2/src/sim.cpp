#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#ifdef _MSC_VER
#include <nmmintrin.h>
#endif
#include "sim.hpp"

namespace py = pybind11;

#ifdef _WIN64
__inline long long popcntll(unsigned long long X)
{
    return _mm_popcnt_u64(X);
}
#else
__inline long long popcntll(unsigned long long X)
{
    return __builtin_popcountll(X);
}
#endif

uint32_t py_popcount(py::array_t<unsigned long long> pyquery)
{
    auto query = pyquery.unchecked<1>();
    uint32_t qcount = 0;
    for (ssize_t i = 0; i < query.shape(0); i++)
            qcount += popcntll(query(i));
    return qcount;
}

__inline float tanimoto_coeff(uint32_t int_count, uint32_t qcount, uint32_t ocount)
{
    float coeff = 0.0;
    coeff = qcount + ocount - int_count;
    if (coeff != 0.0)
        coeff = int_count / coeff;
    return coeff;
}

__inline float substruct_coeff(uint32_t rel_co_count, uint32_t int_count)
{
    float coeff = 0.0;
    coeff = rel_co_count + int_count;
    if (coeff != 0.0)
        coeff = int_count / coeff;
    return coeff;
}

py::array_t<uint32_t> _substructure_search(py::array_t<unsigned long long> pyquery,
                                           py::array_t<unsigned long long> pydb,
                                           float threshold,
                                           uint32_t i_start,
                                           uint32_t i_end)
{
    // release the GIL
    py::gil_scoped_release release;

    auto query = pyquery.unchecked<1>();
    auto db = pydb.unchecked<2>();

    // initial similarity result array size, allocate memory for results
    uint32_t subsres_length = 256;
    uint32_t *results = (uint32_t *) malloc(subsres_length * sizeof(uint32_t));

    size_t qshape = query.shape(0);
    uint8_t popcntidx = qshape - 1;

    uint32_t int_count = 0;
    uint32_t rel_co_count = 0;
    float coeff = 0.0;
    uint32_t total_subs = 0;
    uint32_t i = i_start;
    while (i_end > i)
    {
        // calc count for intersection and relative complement
        for (size_t j = 1; j < popcntidx; j++)
        {
            int_count += popcntll(query(j) & db(i, j));
            rel_co_count += popcntll(query(j) & ~db(i, j));
        }
        // calc tversky coeff
        coeff = substruct_coeff(rel_co_count, int_count);

        if (coeff == threshold)
        {
            results[total_subs] = db(i, 0);
            total_subs += 1;
        }

        if (total_subs == subsres_length)
        {
            subsres_length *= 2;
            results = (uint32_t *) realloc(results, subsres_length * sizeof(uint32_t));
        }
        //  reset values for next fp
        int_count = 0;
        rel_co_count = 0;
        i++;
    }

    // acquire the GIL
    py::gil_scoped_acquire acquire;
    auto subs = py::array_t<uint32_t>(total_subs, results);
    return subs;
}

py::array_t<Result> _similarity_search(py::array_t<unsigned long long> pyquery,
                                       py::array_t<unsigned long long> pydb,
                                       float threshold,
                                       uint32_t i_start,
                                       uint32_t i_end)
{
    // release the GIL
    py::gil_scoped_release release;

    // direct access to np arrays without checks
    auto query = pyquery.unchecked<1>();
    auto db = pydb.unchecked<2>();

    // initial similarity result array size, allocate memory for results
    uint32_t simres_length = 256;
    Result *results = (Result *)malloc(simres_length * sizeof(Result));

    size_t qshape = query.shape(0);
    uint8_t popcntidx = qshape - 1;

    uint32_t int_count = 0;
    uint32_t total_sims = 0;
    float coeff = 0.0;
    uint32_t i = i_start;
    while (i_end > i)
    {
        for (size_t j = 1; j < popcntidx; j++)
            int_count += popcntll(query(j) & db(i, j));
        coeff = tanimoto_coeff(int_count, query(popcntidx), db(i, popcntidx));

        if (coeff >= threshold)
        {
            results[total_sims].mol_id = db(i, 0);
            results[total_sims].coeff = coeff;
            total_sims += 1;
        }

        if (total_sims == simres_length)
        {
            // reallocate memory
            simres_length *= 2;
            results = (Result *)realloc(results, simres_length * sizeof(Result));
        }
        int_count = 0;
        i++;
    }

    // acquire the GIL
    py::gil_scoped_acquire acquire;
    auto sims = py::array_t<Result>(total_sims, results);
    return sims;
}
