import concurrent.futures as cf
from .FPSim2lib import on_disk_search, _similarity_search, _substructure_search, get_bounds_range
from .io import load_fps, load_query, S_INDEXS
import tables as tb
import numpy as np


class FPSim2Engine:

    fp_filename = None
    fp_type = None
    fp_params = None
    rdkit_ver = None
    fps = None

    def __init__(self, fp_filename, in_memory_fps=True, fps_sort=False):
        self.fp_filename = fp_filename
        with tb.open_file(fp_filename, mode='r') as fp_file:
            self.fp_type = fp_file.root.config[0]
            self.fp_params = fp_file.root.config[1]
            self.rdkit_ver = fp_file.root.config[2]
        if in_memory_fps:
            self.load_fps(fps_sort)        

    def load_fps(self, fps_sort=False):
        self.fps = load_fps(self.fp_filename, fps_sort)

    def _preflight(self, query_string, count_ranges, threshold, s_index):
        query = load_query(query_string, self.fp_filename)
        fp_range = get_bounds_range(query, count_ranges, threshold, S_INDEXS[s_index])
        return query, fp_range

    def _parallel_run(self, query, search_func, executor, fp_range, n_workers, threshold, chunk_size, s_index, on_disk):
        i_start = fp_range[0]
        i_end = fp_range[1]
        results = []
        with executor(max_workers=n_workers) as exe:
            if not on_disk:
                chunk_size = int((i_end - i_start) / n_workers)
            c_indexes = [[x, x + chunk_size] for x in range(i_start, i_end, chunk_size)]
            c_indexes[-1][1] = i_end
            if on_disk:
                future_ss = {exe.submit(search_func, query, self.fp_filename, indexes, threshold, S_INDEXS[s_index]): 
                    c_id for c_id, indexes in enumerate(c_indexes)}
            else:
                future_ss = {exe.submit(search_func, query, self.fps.fps, threshold, chunk_idx[0], chunk_idx[1]): 
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
            with tb.open_file(self.fp_filename, mode='r') as fp_file:
                count_ranges = fp_file.root.config[3]
        else:
            count_ranges = self.fps.count_ranges

        if s_index == 'tanimoto':
            empty_np = np.ndarray((0,), dtype=[('mol_id','u8'), ('coeff','f4')])
        else:
            empty_np = np.ndarray((0,), dtype='<u8')

        query, fp_range = self._preflight(query_string, count_ranges, threshold, s_index)
        if fp_range:
            if n_threads == 1:
                np_res = search_func(query, fps.fps, threshold, i_start, i_end)
                if s_index != 'substructure':
                    np_res[::-1].sort(order='coeff')
            else:
                results = self._parallel_run(query, 
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
                    if s_index != 'substructure':
                        np_res[::-1].sort(order='coeff')
                else:
                    np_res = empty_np
        else:
            np_res = empty_np
        return np_res

    def similarity(self, query_string, theshold, n_workers=1):
        if not self.fps:
            raise Exception('Load the fingerprints into memory before running a in memory search')
        return self._base_search(query_string, theshold, _similarity_search, 0, 'tanimoto', False, cf.ThreadPoolExecutor, n_workers)

    def on_disk_similarity(self, query_string, theshold, n_workers=1, chunk_size=250000):
        return self._base_search(query_string, theshold, on_disk_search, chunk_size, 'tanimoto', True, cf.ProcessPoolExecutor, n_workers)

    def substructure(self, query_string, n_workers=1):
        if not self.fps:
            raise Exception('Load the fingerprints into memory before running a in memory search')
        return self._base_search(query_string, 1.0, _substructure_search, 0, 'substructure', False, cf.ThreadPoolExecutor, n_workers)

    def on_disk_substructure(self, query_string, n_workers=1, chunk_size=250000):
        return self._base_search(query_string, 1.0, on_disk_search, chunk_size, 'substructure', True, cf.ProcessPoolExecutor, n_workers)
