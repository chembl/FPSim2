import multiprocessing as mp
import concurrent.futures as cf
from .FPSim2lib import similarity_search, in_memory_ss
from .FPSim2lib import get_bounds_range
from .io import tables, load_query, COEFFS
from operator import itemgetter
import numpy as np
import time


def run_search(query, fp_filename, threshold=0.7, coeff='tanimoto', chunk_size=1000000, n_threads=mp.cpu_count()):
    with tables.open_file(fp_filename, mode='r') as fp_file:
        n_mols = fp_file.root.fps.shape[0]
        fp_tpye = fp_file.root.config[0]

    query = load_query(query, fp_filename)

    if coeff == 'substructure':
         # if substructure automatically set threshold to 1.0
        threshold = 1.0
        if fp_tpye != 'RDKPatternFingerprint':
            print('Warning: Running a substructure search with {} fingerprints. '
                'Consider using RDKPatternFingerprint'.format(fp_tpye))

    fp_range = get_bounds_range(query, fps, threshold, COEFFS[coeff])
    if not fp_range:
        return []
    else:
        i_start = fp_range[0]
        i_end = fp_range[1]

    # chunkify
    c_indexes = ((x, x + chunk_size) for x in range(i_start, i_end, chunk_size))
    results = []
    with cf.ProcessPoolExecutor(max_workers=n_threads) as ppe:
        future_ss = {ppe.submit(similarity_search, query, fp_filename, indexes, threshold, COEFFS[coeff]): 
                        c_id for c_id, indexes in enumerate(c_indexes)}
        for future in cf.as_completed(future_ss):
            m = future_ss[future]
            try:
                res = future.result()
                if res.shape[0] != 0:
                    results.append(res)
            except Exception as e:
                print('Chunk {} thread died: '.format(m), e)
    np_res = np.concatenate(results)
    np_res[::-1].sort(order='coeff')
    return np_res


def run_in_memory_search(query, fps, threshold=0.7, coeff='tanimoto', n_threads=mp.cpu_count()):
    if coeff == 'substructure':
        threshold = 1.0
    fp_range = get_bounds_range(query, fps, threshold, COEFFS[coeff])
    if not fp_range:
        return []
    else:
        i_start = fp_range[0]
        i_end = fp_range[1]

    if n_threads == 1:
        np_res = in_memory_ss(query, fps[0], threshold, COEFFS[coeff], i_start, i_end)
    else:
        results = []
        with cf.ThreadPoolExecutor(max_workers=n_threads) as tpe:
            chunk_size = int((i_end - i_start) / n_threads)
            c_indexes = ((x, x + chunk_size) for x in range(i_start, i_end, chunk_size))
            future_ss = {tpe.submit(in_memory_ss, query, fps[0], threshold, COEFFS[coeff], chunk_idx[0], chunk_idx[1]): 
                            c_id for c_id, chunk_idx in enumerate(c_indexes)}
            for future in cf.as_completed(future_ss):
                m = future_ss[future]
                try:
                    res = future.result()
                    if res.shape[0] != 0:
                        results.append(res)
                except Exception as e:
                    print('Chunk {} thread died: '.format(m), e)
        np_res = np.concatenate(results)
    np_res[::-1].sort(order='coeff')
    return np_res
