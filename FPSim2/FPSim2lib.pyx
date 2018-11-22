# distutils: language = c
# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport malloc, realloc, free
import tables as tb


# include CPU popcnt functions built in GCC for all posix systems
# using -march=native GCC flag will use best CPU instruction available
cdef extern int __builtin_popcountll(unsigned long long) nogil


cdef struct Result:
    uint64_t mol_id
    float coeff


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline double _substruct_coeff(uint32_t fp_subs, uint32_t int_count) nogil:
    cdef double s_coeff
    s_coeff = fp_subs + int_count
    if s_coeff != 0.0:
        s_coeff = int_count / s_coeff
    return s_coeff


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline double _tanimoto_coeff(uint32_t int_count, uint32_t count_query, uint32_t count_other) nogil:
    cdef double t_coeff = 0.0
    t_coeff = count_query + count_other - int_count
    if t_coeff != 0.0:
        t_coeff = int_count / t_coeff
    return t_coeff


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef _similarity_search(uint64_t[:] query, uint64_t[:, :] fps, uint32_t[:] popcnt, double threshold, int coeff_func, int i_start, int i_end):

    cdef int i
    cdef int j
    cdef uint32_t int_count = 0
    cdef uint32_t rel_co_count = 0
    cdef uint32_t query_count = 0
    cdef double coeff = 0.0
    cdef int total_sims = 0
    cdef int simres_length = 256

    # allocate number * sizeof(Result) bytes of memory
    cdef Result *results = <Result *> malloc(simres_length * sizeof(Result))

    with nogil:
        # precalc query popcount
        for j in range(query.shape[0]):
            query_count += __builtin_popcountll(query[j])

        for i in range(i_start, i_end):
            for j in range(0, query.shape[0], 4):
                # Use __builtin_popcountll for unsigned 64-bit integers (fps j+ 1 in fps to skip the mol_id)
                # equivalent to https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-builtin.cpp#L23
                int_count += __builtin_popcountll(fps[i, j + 1] & query[j])
                int_count += __builtin_popcountll(fps[i, j + 2] & query[j + 1])
                int_count += __builtin_popcountll(fps[i, j + 3] & query[j + 2])
                int_count += __builtin_popcountll(fps[i, j + 4] & query[j + 3])

                if coeff_func == 2:
                    rel_co_count +=  __builtin_popcountll(query[j] & ~fps[i, j + 1])
                    rel_co_count +=  __builtin_popcountll(query[j + 1] & ~fps[i, j + 2])
                    rel_co_count +=  __builtin_popcountll(query[j + 2] & ~fps[i, j + 3])
                    rel_co_count +=  __builtin_popcountll(query[j + 3] & ~fps[i, j + 4])

            # tanimoto
            if coeff_func == 0:
                coeff = _tanimoto_coeff(int_count, query_count, popcnt[i])
            # substruct (tversky a=1, b=0 eq)
            elif coeff_func == 2:
                coeff = _substruct_coeff(rel_co_count, int_count)

            if coeff >= threshold:
                results[total_sims].mol_id = fps[i][0]
                results[total_sims].coeff = coeff
                total_sims += 1

            if total_sims == simres_length:
                simres_length *= 2
                # reallocating memory
                results = <Result *> realloc(results, simres_length * sizeof(Result))

            # reset values for next fp
            int_count = 0
            rel_co_count = 0

    # this is happening inside the GIL
    cdef np.ndarray np_results = np.ndarray((total_sims,), dtype=[('mol_id','u8'), ('coeff','f4')])
    for i in range(total_sims):
        np_results[i][0] = results[i].mol_id
        np_results[i][1] = results[i].coeff

    # free manually allocated memory and return the results
    free(results)
    return np_results


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef int py_popcount(query):
    cdef int query_count = 0
    cdef int j
    for j in range(query.shape[0]):
        query_count += __builtin_popcountll(query[j])
    return query_count


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef get_bounds_range(query, ranges, threshold, coeff):
    cdef float max_sim

    query_count = py_popcount(query)
    range_to_keep = []

    for count, c_range in ranges:
        # tanimoto
        if coeff == 0:
            max_sim = min(query_count, count) / max(query_count, count)
        # substructure
        elif coeff == 2:
            max_sim = min(query_count, count) / query_count
        else:
            break

        if max_sim >= threshold:
            range_to_keep.append(c_range)

    if range_to_keep:
        range_to_keep = (range_to_keep[0][0], range_to_keep[len(range_to_keep)-1][1])

    return range_to_keep


def similarity_search(query, fp_filename, chunk_indexes, threshold, coeff):
    with tb.open_file(fp_filename, mode='r') as fp_file:
        fps = fp_file.root.fps[chunk_indexes[0]:chunk_indexes[1]]

    fnames = [x for x in fps.dtype.names[0:-1]]
    # numpy 1.16 returns a view, not a copy
    popcnt = fps[['popcnt']].view('<u4')
    fps = fps[fnames]
    num_fields = len(fps[0])
    fps = fps.view('<u8')
    fps = fps.reshape(int(fps.size / num_fields), num_fields)

    res = _similarity_search(query, fps, popcnt, threshold, coeff, 0, fps.shape[0])
    return res
