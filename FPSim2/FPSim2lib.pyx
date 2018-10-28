# distutils: language = c
# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport malloc, realloc
from .io import tables as tb
import time

# include CPU popcnt functions built in GCC for all posix systems
# using -march=native GCC flag will use best CPU instruction available
cdef extern int __builtin_popcountll(unsigned long long) nogil


cdef packed struct Result:
    uint64_t mol_id
    float coeff


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline double _substruct_coeff(int fp_subs, int int_count) nogil:
    cdef double s_coeff
    s_coeff = fp_subs + int_count
    if s_coeff != 0.0:
        s_coeff = int_count / s_coeff
    return s_coeff


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline double _tanimoto_coeff(int int_count, int count_query, int count_other) nogil:
    cdef double t_coeff = 0.0
    t_coeff = count_query + count_other - int_count
    if t_coeff != 0.0:
        t_coeff = int_count / t_coeff
    return t_coeff


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef _similarity_search(uint64_t[:, :] query, uint64_t[:, :] fps, double threshold, int coeff_func, int i_start, int i_end):

    cdef int i
    cdef int j
    cdef int int_count = 0
    cdef int rel_co_count = 0
    cdef int query_count = 0
    cdef double coeff
    cdef int total_sims = 0
    cdef int simres_length = 256

    # allocate number * sizeof(Result) bytes of memory
    cdef Result *results = <Result *> malloc(simres_length * sizeof(Result))

    with nogil:
        # precalc query popcount
        for j in range(query.shape[1]):
            query_count += __builtin_popcountll(query[0, j])

        for i in range(i_start, i_end):
            for j in range(0, query.shape[1], 4):
                # Use __builtin_popcountll for unsigned 64-bit integers (fps j+ 1 in fps to skip the mol_id)
                # equivalent to https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-builtin.cpp#L23
                int_count += __builtin_popcountll(fps[i, j + 1] & query[0, j])
                int_count += __builtin_popcountll(fps[i, j + 2] & query[0, j + 1])
                int_count += __builtin_popcountll(fps[i, j + 3] & query[0, j + 2])
                int_count += __builtin_popcountll(fps[i, j + 4] & query[0, j + 3])

                if coeff_func == 2:
                    rel_co_count +=  __builtin_popcountll(query[0, j] & ~fps[i, j + 1])
                    rel_co_count +=  __builtin_popcountll(query[0, j + 1] & ~fps[i, j + 2])
                    rel_co_count +=  __builtin_popcountll(query[0, j + 2] & ~fps[i, j + 3])
                    rel_co_count +=  __builtin_popcountll(query[0, j + 3] & ~fps[i, j + 4])

            # tanimoto
            if coeff_func == 0:
                coeff = _tanimoto_coeff(int_count, query_count, fps[i, query.shape[1] + 1])
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

    # create a memory view
    view = <Result[:total_sims]> results
    return np.asarray(view)


cpdef int py_popcount(query):
    cdef int query_count = 0
    cdef int j
    for j in range(query.shape[1]):
        query_count += __builtin_popcountll(query[0, j])
    return query_count


@cython.boundscheck(False)
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
            max_sim = min(query_count, count) / count
        else:
            break

        if max_sim >= threshold:
            range_to_keep.append(c_range)

    if range_to_keep:
        range_to_keep = (range_to_keep[0][0], range_to_keep[-1][1])

    return range_to_keep


def similarity_search(query, fp_filename, chunk_indexes, threshold, coeff):
    with tb.open_file(fp_filename, mode='r') as fp_file:
        fps = fp_file.root.fps[chunk_indexes[0]:chunk_indexes[1]]
    res = _similarity_search(query, fps, threshold, coeff, 0, fps.shape[0])
    return res
