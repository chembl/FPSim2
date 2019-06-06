#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#if defined(_MSC_VER)
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

float tanimoto_coeff(uint32_t int_count, uint32_t count_query, uint32_t count_other)
{
    float t_coeff = 0.0;
    t_coeff = count_query + count_other - int_count;
    if (t_coeff != 0.0)
        t_coeff = int_count / t_coeff;
    return t_coeff;
}

float substruct_coeff(uint32_t rel_co_count, uint32_t int_count)
{
    float s_coeff = 0.0;
    s_coeff = rel_co_count + int_count;
    if (s_coeff != 0.0)
        s_coeff = int_count / s_coeff;
    return s_coeff;
}

uint32_t py_popcount(py::array_t<unsigned long long> query)
{
    auto qbuff = query.request();

    // release the GIL
    py::gil_scoped_release release;

    unsigned long long *qptr = (unsigned long long *)qbuff.ptr;
    int qshape = qbuff.shape[0];
    uint32_t query_count = 0;
    for (size_t i = 0; i < qshape; i++)
            query_count += popcntll(qptr[i]);

    // acquire the GIL
    py::gil_scoped_acquire acquire;
    return query_count;
}

py::array_t<uint32_t> _substructure_search(py::array_t<unsigned long long> query,
                                           py::array_t<unsigned long long> db,
                                           float threshold,
                                           uint32_t i_start,
                                           uint32_t i_end)
{

    auto dbbuff = db.request(), qbuff = query.request();

    // release the GIL
    py::gil_scoped_release release;

    // pointers to buffers
    unsigned long long *dbptr = (unsigned long long *)dbbuff.ptr,
                       *qptr = (unsigned long long *)qbuff.ptr;

    // initial similarity result array size
    uint32_t simres_length = 256;
    int dbshapeY = dbbuff.shape[1];
    int qshape = qbuff.shape[0];

    uint32_t *results = (uint32_t *) malloc(simres_length * sizeof(uint32_t));

    uint32_t int_count = 0;
    uint32_t rel_co_count = 0;
    float coeff = 0.0;
    uint32_t total_sims = 0;
    uint32_t i = i_start;
    while (i_end > i)
    {
        // calc count for intersection and relative complement
        for (size_t j = 0; j < qshape; j++)
            int_count += popcntll(qptr[j] & dbptr[i * dbshapeY + j + 1]);
        for (size_t j = 0; j < qshape; j++)
            rel_co_count += popcntll(qptr[j] & ~dbptr[i * dbshapeY + j + 1]);
        // calc tversky coeff
        coeff = substruct_coeff(rel_co_count, int_count);

        if (coeff == threshold)
        {
            results[total_sims] = dbptr[i * dbshapeY];
            total_sims += 1;
        }

        if (total_sims == simres_length)
        {
            simres_length *= 2;
            results = (uint32_t *) realloc(results, simres_length * sizeof(uint32_t));
        }
        //  reset values for next fp
        int_count = 0;
        rel_co_count = 0;
        i++;
    }

    // acquire the GIL
    py::gil_scoped_acquire acquire;

    // we can create a result numpy array
    auto subs = py::array_t<uint32_t>(total_sims);
    py::buffer_info bufsubs = subs.request();
    uint32_t *ptrsubs = (uint32_t *)bufsubs.ptr;

    for (size_t i = 0; i < total_sims; i++)
        ptrsubs[i] = results[i];

    free(results);
    return subs;
}

py::array_t<Result> _similarity_search(py::array_t<unsigned long long> query,
                                       py::array_t<unsigned long long> db,
                                       float threshold,
                                       uint32_t i_start,
                                       uint32_t i_end)
{

    auto dbbuff = db.request(), qbuff = query.request();

    // release the GIL
    py::gil_scoped_release release;

    // pointers to buffers
    unsigned long long *dbptr = (unsigned long long *)dbbuff.ptr,
                       *qptr = (unsigned long long *)qbuff.ptr;

    // initial similarity result array size
    uint32_t simres_length = 256;
    int dbshapeY = dbbuff.shape[1];
    int qshape = qbuff.shape[0];

    // allocate number * sizeof(Result) bytes of memory
    Result *results = (Result *)malloc(simres_length * sizeof(Result));

    // calc query molecule popcount
    uint32_t qcount = 0;
    for (size_t i = 0; i < qshape; i++)
        qcount += popcntll(qptr[i]);

    uint32_t int_count = 0;
    uint32_t total_sims = 0;
    float coeff = 0.0;
    uint32_t i = i_start;
    while (i_end > i)
    {
        for (size_t j = 0; j < qshape; j++)
            int_count += popcntll(qptr[j] & dbptr[i * dbshapeY + j + 1]);
        coeff = tanimoto_coeff(int_count, qcount, dbptr[i * dbshapeY + qshape + 1]);

        if (coeff >= threshold)
        {
            results[total_sims].mol_id = dbptr[i * dbshapeY];
            results[total_sims].coeff = coeff;
            total_sims += 1;
        }

        if (total_sims == simres_length)
        {
            simres_length *= 2;
            // reallocating memory
            results = (Result *)realloc(results, simres_length * sizeof(Result));
        }
        int_count = 0;
        i++;
    }

    // acquire the GIL
    py::gil_scoped_acquire acquire;

    auto sims = py::array_t<Result>(total_sims);
    py::buffer_info bufsims = sims.request();
    Result *ptrsims = (Result *)bufsims.ptr;

    for (size_t i = 0; i < total_sims; i++)
    {
        ptrsims[i].mol_id = results[i].mol_id;
        ptrsims[i].coeff = results[i].coeff;
    }
    free(results);
    return sims;
}
