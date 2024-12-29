from FPSim2.io.backends.pytables import create_db_file, merge_db_files
import concurrent.futures as cf
from typing import List, Tuple
from itertools import islice
import argparse
import json
import os


def read_chunk(filename, start_row, end_row):
    with open(filename, "r") as f:
        # works with both SMILES and CXMILES
        chunk = islice(f, start_row, end_row)
        for line in chunk:
            try:
                mol = line.strip().rsplit(None, 1)
                if len(mol) == 2:
                    yield mol
            except ValueError:
                continue

def worker(args):
    mols_source, chunk_range, fp_type, fp_params = args
    out_file = f"temp_chunk_{chunk_range[0]}.h5"

    rows = read_chunk(mols_source, chunk_range[0], chunk_range[1])
    create_db_file(
        rows,
        out_file,
        mol_format="smiles",
        fp_type=fp_type,
        fp_params=fp_params,
        sort_by_popcnt=False,
    )
    return out_file


def calculate_chunks(
    total_rows: int, num_processes: int, m: int = 16
) -> List[Tuple[int, int]]:
    n = num_processes * m
    rows_per_chunk = total_rows // n
    chunks = []
    start = 0

    for i in range(n):
        if i == n - 1:
            end = total_rows
        else:
            end = start + rows_per_chunk
        chunks.append((start, end))
        start = end
    return chunks


def count_rows(filename, chunk_size=65536):
    with open(filename, "rb") as f:
        chunks = iter(lambda: f.read(chunk_size), b"")
        newlines = sum(chunk.count(b"\n") for chunk in chunks)
        # If file doesn't end with newline, add 1 for the last line
        f.seek(-1, os.SEEK_END)
        last_char = f.read(1)
        return newlines + (0 if last_char == b"\n" else 1)


def create_db_file_parallel(smi_file, out_file, fp_type, fp_params, num_processes):
    total_mols = count_rows(smi_file)
    chunks = calculate_chunks(total_mols, num_processes)
    with cf.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(
                worker,
                (smi_file, chunk, fp_type, fp_params),
            )
            for chunk in chunks
        ]
        tmp_filenames = []
        for future in futures:
            tmp_filenames.append(future.result())

    merge_db_files(tmp_filenames, out_file, sort_by_popcnt=True)

    # Clean up temporary files
    for temp_file in tmp_filenames:
        os.remove(temp_file)


def main():
    parser = argparse.ArgumentParser(
        description="Create fingerprint database in parallel"
    )
    parser.add_argument("input_file", help="Input SMILES file")
    parser.add_argument("output_file", help="Output H5 database file")
    parser.add_argument(
        "--fp_type", default="Morgan", help="Fingerprint type (default: Morgan)"
    )

    parser.add_argument(
        "--fp_params",
        type=json.loads,
        default='{"radius": 2, "fpSize": 256}',
        help='Fingerprint parameters as JSON string (default: {"radius": 2, "fpSize": 256})',
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=8,
        help="Number of parallel processes (default: 8)",
    )

    args = parser.parse_args()

    if not args.input_file.endswith(".smi"):
        parser.error("Input file must have '.smi' SMILES file")

    create_db_file_parallel(
        args.input_file,
        args.output_file,
        args.fp_type,
        args.fp_params,
        args.processes,
    )
