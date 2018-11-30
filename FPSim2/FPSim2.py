import multiprocessing as mp
import concurrent.futures as cf
from .FPSim2lib import similarity_search, _similarity_search, get_bounds_range
from .io import load_query, COEFFS
import tables as tb
import numpy as np


def run_search(query, fp_filename, threshold=0.7, coeff='tanimoto', chunk_size=1000000, db_sorted=False, n_threads=mp.cpu_count()):
    with tb.open_file(fp_filename, mode='r') as fp_file:
        n_mols = fp_file.root.fps.shape[0]
        fp_tpye = fp_file.root.config[0]
        if db_sorted:
            count_ranges = fp_file.root.config[3]

    query = load_query(query, fp_filename)

    if coeff == 'substructure':
         # if substructure automatically set threshold to 1.0
        threshold = 1.0
        if fp_tpye != 'RDKPatternFingerprint':
            print('Warning: Running a substructure search with {} fingerprints. '
                'Consider using RDKPatternFingerprint'.format(fp_tpye))

    if db_sorted:
        fp_range = get_bounds_range(query, count_ranges, threshold, COEFFS[coeff])
        if not fp_range:
            return np.asarray([])
        else:
            i_start = fp_range[0]
            i_end = fp_range[1]
    else:
        i_start = 0
        i_end = n_mols - 1

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
            except ValueError:
                pass
            except Exception as e:
                print('Chunk {} thread died: '.format(m), e)
    if results:
        np_res = np.concatenate(results)
        np_res[::-1].sort(order='coeff')
    else:
        np_res = np.asarray(results)
    return np_res


def run_in_memory_search(query, fps, threshold=0.7, coeff='tanimoto', n_threads=mp.cpu_count()):
    if coeff == 'substructure':
        threshold = 1.0
    fp_range = get_bounds_range(query, fps.count_ranges, threshold, COEFFS[coeff])
    if not fp_range:
        return np.asarray([])
    else:
        i_start = fp_range[0]
        i_end = fp_range[1]

    if n_threads == 1:
        np_res = _similarity_search(query, fps.fps, threshold, COEFFS[coeff], i_start, i_end)
        np_res[::-1].sort(order='coeff')
        return np_res
    else:
        results = []
        with cf.ThreadPoolExecutor(max_workers=n_threads) as tpe:
            chunk_size = int((i_end - i_start) / n_threads)
            c_indexes = [[x, x + chunk_size] for x in range(i_start, i_end, chunk_size)]
            c_indexes[-1][1] = i_end
            future_ss = {tpe.submit(_similarity_search, query, fps.fps, threshold, COEFFS[coeff], chunk_idx[0], chunk_idx[1]): 
                            c_id for c_id, chunk_idx in enumerate(c_indexes)}
            for future in cf.as_completed(future_ss):
                m = future_ss[future]
                try:
                    res = future.result()
                    if res.shape[0] != 0:
                        results.append(res)
                except ValueError:
                    pass
                except Exception as e:
                    print('Chunk {} thread died: '.format(m), e)
        if results:
            np_res = np.concatenate(results)
            np_res[::-1].sort(order='coeff')
        else:
            np_res = np.asarray(results)
    return np_res
