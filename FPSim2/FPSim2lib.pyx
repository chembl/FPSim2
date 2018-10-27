# distutils: language = c++
# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from libc.stdint cimport uint32_t, uint64_t
from libcpp.list cimport list as cpplist
from libcpp.vector cimport vector
from .io import tables as tb
import time

# include CPU popcnt functions built in GCC for all posix systems
# using -march=native GCC flag will use best CPU instruction available
cdef extern int __builtin_popcountll(unsigned long long) nogil


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
cdef inline double _dice_coeff(int int_count, int query_count, int other_count) nogil:
    cdef double d_coeff
    d_coeff = query_count + other_count
    if d_coeff != 0.0:
        d_coeff = int_count * 2 / d_coeff
    return d_coeff


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
cpdef _similarity_search(uint64_t[:, :] query, uint64_t[:, :] fps, double threshold, int coeff_func):

    cdef int i
    cdef int j
    cdef int int_count = 0
    cdef int rel_co_count = 0
    cdef int query_count = 0
    cdef double coeff

    cdef vector[double] temp_scores
    cdef vector[uint64_t] temp_ids

    with nogil:

        # precalc query popcount
        for j in range(query.shape[1]):
            query_count += __builtin_popcountll(query[0, j])

        for i in range(fps.shape[0]):
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
            # dice
            elif coeff_func == 1:
                coeff = _dice_coeff(int_count, query_count, fps[i, query.shape[1] + 1])
            # substruct (tversky a=1, b=0 eq)
            elif coeff_func == 2:
                coeff = _substruct_coeff(rel_co_count, int_count)

            if coeff >= threshold:
                temp_scores.push_back(coeff)
                temp_ids.push_back(fps[i][0])

            # reset values for next fp
            int_count = 0
            rel_co_count = 0

    # inside the GIL :(
    cdef np.ndarray results = np.ndarray((temp_scores.size(),), dtype=[('mol_id','i8'), ('coeff','f4')])
    for i in range(temp_scores.size()):
        results[i][0] = temp_ids.back()
        results[i][1] = temp_scores.back()
        temp_scores.pop_back()
        temp_ids.pop_back()

    return results


cpdef int py_popcount(query):
    cdef int query_count = 0
    cdef int j
    for j in range(query.shape[1]):
        query_count += __builtin_popcountll(query[0, j])
    return query_count


cpdef filter_by_bound(query, fps, threshold, coeff):
    query_count = py_popcount(query)
    cdef int i
    cdef int j
    cdef float a
    cdef int start = 0

    range_to_keep = []
    # tanimoto
    if coeff == 0:
        for i, j in enumerate(fps[1][0]):
            a = min(query_count, j) / max(query_count, j)
            if start == 0:
                start = 1
                if a >= threshold:
                    range_to_keep.append(fps[1][1][i])
            else:
                if a >= threshold:
                    range_to_keep.append(fps[1][1][i+1])
        fps = fps[0][range_to_keep[0]:range_to_keep[-1]]
    # substruct
    elif coeff == 2:
        for i in range(1, query.shape[1]*64 + 1):
            a = min(query_count, i) / i
            if a >= threshold:
                range_to_keep.append(i)
        fps = fps[np.in1d(fps[:,-1], range_to_keep)]
    return fps


def similarity_search(query, fp_filename, chunk_indexes, threshold=0.7, coeff=0):
    with tb.open_file(fp_filename, mode='r') as fp_file:
        fps = fp_file.root.fps[chunk_indexes[0]:chunk_indexes[1]]
    res = _similarity_search(query, fps, threshold, coeff)
    return res


def in_memory_ss(query, fps, threshold=0.7, coeff=0):
    res = _similarity_search(query, fps, threshold, coeff)
    return res
