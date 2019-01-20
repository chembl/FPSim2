import multiprocessing as mp
import concurrent.futures as cf
from .FPSim2lib import run_search, _similarity_search, _substructure_search, get_bounds_range
from .io import load_query, COEFFS
import tables as tb
import numpy as np


def on_disk_search(query, fp_filename, threshold=0.7, coeff='tanimoto', chunk_size=1000000, n_processes=mp.cpu_count()):
    """ Run a on disk search.
    
    :param query: Query molecule. SMILES, molblock or InChi formats accepted.
    :param fp_filename: FPs filename.
    :param threshold: Threshold for similarity.
    :param coeff: Coefficient. 'tanimoto' or 'substructure'.
    :param chunk_size: Chunk size.
    :param n_threads: Number of threads used to do the search.
    :return: Numpy structured array with mol_id and coeff for each match.
    """
    with tb.open_file(fp_filename, mode='r') as fp_file:
        fp_tpye = fp_file.root.config[0]
        count_ranges = fp_file.root.config[3]

    query = load_query(query, fp_filename)

    if coeff == 'substructure':
        empty_res = np.ndarray((0,), dtype='<u8')
        threshold = 1.0
        if fp_tpye != 'RDKPatternFingerprint':
            print('Warning: Running a substructure search with {} fingerprints. '
                'Consider using RDKPatternFingerprint'.format(fp_tpye))
    else:
        empty_res = np.ndarray((0,), dtype=[('mol_id','u8'), ('coeff','f4')])

    fp_range = get_bounds_range(query, count_ranges, threshold, COEFFS[coeff])
    if not fp_range:
        return empty_res
    else:
        i_start = fp_range[0]
        i_end = fp_range[1]

    c_indexes = ((x, x + chunk_size) for x in range(i_start, i_end, chunk_size))
    results = []
    with cf.ProcessPoolExecutor(max_workers=n_processes) as ppe:
        future_ss = {ppe.submit(run_search, query, fp_filename, indexes, threshold, COEFFS[coeff]): 
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
                print('Chunk {} process died: '.format(m), e)
    if results:
        np_res = np.concatenate(results)
        if coeff != 'substructure':
            np_res[::-1].sort(order='coeff')
    else:
        np_res = empty_res
    return np_res


def search(query, fps, threshold=0.7, coeff='tanimoto', n_threads=1):
    """ Run a in memory search
    
    :param query: Query molecule. Preloaded using load_query function.
    :param fps: Fingerprints. Loaded using load_fps function.
    :param threshold: Threshold for similarity.
    :param coeff: Coefficient. Use 'tanimoto' or 'substructure'.
    :param n_threads: Number of threads used to do the search.
    :return: Numpy structured array with mol_id and coeff for each match.
    """
    if coeff == 'substructure':
        empty_res = np.ndarray((0,), dtype='<u8')
        threshold = 1.0
        search_func = _substructure_search
    else:
        empty_res = np.ndarray((0,), dtype=[('mol_id','u8'), ('coeff','f4')])
        search_func = _similarity_search

    fp_range = get_bounds_range(query, fps.count_ranges, threshold, COEFFS[coeff])
    if not fp_range:
        return empty_res
    else:
        i_start = fp_range[0]
        i_end = fp_range[1]

    if n_threads == 1:
        np_res = search_func(query, fps.fps, threshold, i_start, i_end)
        if coeff == 'substructure':
            np_res[::-1].sort(order='coeff')
        return np_res
    else:
        results = []
        with cf.ThreadPoolExecutor(max_workers=n_threads) as tpe:
            chunk_size = int((i_end - i_start) / n_threads)
            c_indexes = [[x, x + chunk_size] for x in range(i_start, i_end, chunk_size)]
            c_indexes[-1][1] = i_end
            future_ss = {tpe.submit(search_func, query, fps.fps, threshold, chunk_idx[0], chunk_idx[1]): 
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
            if coeff != 'substructure':
                np_res[::-1].sort(order='coeff')
        else:
            np_res = empty_res
    return np_res
