import multiprocessing as mp
import concurrent.futures as cf
from .FPSim2lib import run_search, _similarity_search, _substructure_search, get_bounds_range
from .io import load_fps, load_query, S_INDEXS
import tables as tb
import numpy as np


class FPSim2DB:

    fp_filename = None
    fp_type = None
    fp_params = None
    rdkit_ver = None
    fps = None

    def __init__(self, fp_filename, fps_load=True, fps_sort=False):
        self.fp_filename = fp_filename
        with tb.open_file(fp_filename, mode='r') as fp_file:
            self.fp_type = fp_file.root.config[0]
            self.fp_params = fp_file.root.config[1]
            self.rdkit_ver = fp_file.root.config[2]
        if fps_load:
            self.fps = load_fps(self.fp_filename, sort)        

    def _prev(query_string, count_ranges, s_index='tanimoto'):
        query = load_query(query_string, self.fp_filename)
        fp_range = get_bounds_range(query, count_ranges, threshold, S_INDEXS[s_index])
        return query, fp_range

    def _parallel_run(query, search_func, executor, fp_range, n_workers, threshold, chunk_size, s_index, on_disk):
        i_start = fp_range[0]
        i_end = fp_range[1]
        results = []
        with executor(max_workers=n_workers) as tpe:
            if not on_disk:
                chunk_size = int((i_end - i_start) / n_workers)
            c_indexes = [[x, x + chunk_size] for x in range(i_start, i_end, chunk_size)]
            c_indexes[-1][1] = i_end
            if on_disk:
                future_ss = {ppe.submit(search_func, query, self.fp_filename, indexes, threshold, S_INDEXS[s_index]): 
                    c_id for c_id, indexes in enumerate(c_indexes)}
            else:
                future_ss = {tpe.submit(search_func, query, self.fps.fps, threshold, chunk_idx[0], chunk_idx[1]): 
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
                    print('Chunk {} worker died: '.format(m), e)
        return results

    def _base_search(self, query_string, threshold, search_func, chunk_size, s_index, on_disk, executor, n_workers):
        if on_disk:
            with tb.open_file(fp_filename, mode='r') as fp_file:
                count_ranges = fp_file.root.config[3]
        else:
            count_ranges = self.fps.count_ranges

        if s_index == 'tanimoto':
            empty_np = np.ndarray((0,), dtype=[('mol_id','u8'), ('coeff','f4')])
        else:
            empty_np = np.ndarray((0,), dtype='<u8')

        query, fp_range = _prev(query_string, count_ranges, s_index=s_index)
        if fp_range:
            results = _parallel_run(query, 
                                    search_func, 
                                    executor, 
                                    fp_range,
                                    n_workers,
                                    threshold,
                                    chunk_size,
                                    s_index,
                                    on_disk)
            if results:
                np_res = np.concatenate(results)
            else:
                np_res = empty_np
        else:
            np_res = empty_np
        return np_res

    def similarity_search(self, query_string, theshold, n_workers=1):
        return _base_search(query_string, theshold, _similarity_search, 0, 'tanimoto', False, cf.ThreadPoolExecutor, n_workers)

    def on_disk_similarity_search(self, query_string, theshold, n_workers=1, chunk_size=250000):
        return _base_search(query_string, theshold, run_search, chunk_size, 'tanimoto', True, cf.ProcessPoolExecutor, mp.cpu_count())

    def substructure_search(self, query_string, n_workers=1):
        return _base_search(query_string, 1.0, _substructure_search, 0, 'substructure', False, cf.ThreadPoolExecutor, n_workers)

    def on_disk_substructure_search(self, query_string, n_workers=1, chunk_size=250000):
        return _base_search(query_string, 1.0, run_search, chunk_size, 'substructure', True, cf.ProcessPoolExecutor, mp.cpu_count())