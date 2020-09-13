"""command-line driver for FPSim2"""

## MIT License
##
## Copyright (C) 2020 Andrew Dalke, Andrew Dalke Scientific AB, The Chemfp Project

## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.


import sys
import time
import argparse

import numpy as np
import tables as tb

import FPSim2
import FPSim2.io
import FPSim2.io.chem

from typing import Any, Callable, Iterable, Dict, List, Tuple, Union

def die(msg: str):
    sys.stderr.write(f"{msg}\n")
    raise SystemExit(1)

### Convert from command-line strings, and to chemfp-like type parameter encodings

# decorator to add an "encoder" function to a decoder function
def encoder(encoder_func: Callable) -> Callable:
    def set_encoder(f: Callable) -> Callable:
        f.encoder = encoder_func
        return f
    return set_encoder

def positive_int_str(value: int) -> str:
    if not isinstance(value, int):
        raise ValueError("must be an integer")
    if not (value >= 1):
        raise ValueError("must be a positive value")
    return str(value)

@encoder(positive_int_str)
def positive_int(s: str) -> int:
    # Don't do int(s) because that allows "+3" and " 3 ", which I don't want
    if not s.isdigit():
        raise argparse.ArgumentTypeError("must be 1 or greater")
    i = int(s)
    if i == 0:
        raise argparse.ArgumentTypeError("must be 1 or greater")
    return i

def nonnegative_int_str(value: int) -> str:
    if not isinstance(value, int):
        raise ValueError("must be an integer")
    if not (value >= 0):
        raise ValueError("must be a non-negative value")
    return str(value)

@encoder(nonnegative_int_str)
def nonnegative_int(s: str) -> int:
    if not s.isdigit():
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return int(s)

def zero_or_one_str(value: Union[bool, int]) -> str:
    if value is True:
        return "1"
    if value is False:
        return "0"
    if isinstance(value, int) and value in (0, 1):
        return str(value)
    raise ValueError("must be True/1 or False/0")
        

@encoder(zero_or_one_str)
def zero_or_one(s: str) -> int:
    if s == "0":
        return 0
    if s == "1":
        return 1
    raise argparse.ArgumentTypeError("must be 0 or 1")

def nonnegative_float(s: str) -> float:
    try:
        value = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a non-negative float")
    if value < 0.0:
        raise argparse.ArgumentTypeError("must be a non-negative float")
    if value > 1000.0:
        raise argparse.ArgumentTypeError("must not be greater than 1000.0")
    if np.isnan(value):
        raise argparse.ArgumentTypeError("must not be NaN")
    return value
        


##### Convert a (fp_name: str, fp_parms: dict) to a chemfp fingerprint type string

_encoders = {
    "fpSize": positive_int,
    "nBits": positive_int,
    "minPath": nonnegative_int,
    "maxPath": nonnegative_int,
    "nBitsPerHash": nonnegative_int,
    "useHs": zero_or_one,
    "useBondOrder": zero_or_one,
    "branchedPaths": zero_or_one,
    "radius": nonnegative_int,
    "useChirality": zero_or_one,
    "useBondTypes": zero_or_one,
    "useFeatures": zero_or_one,
    "minLength": nonnegative_int,
    "maxLength": nonnegative_int,
    "nBitsPerEntry": positive_int,
    "includeChirality": zero_or_one,
    "use2D": zero_or_one,
    "targetSize": nonnegative_int,
    "atomCounts": zero_or_one,
    "isQuery": zero_or_one,
    "bitFlags": nonnegative_int,
    }
    
class FPType:
    def __init__(
            self,
            fp_type: str,
            fps_typename: str,
            args: List[str],
            optional_args: List[Tuple[str, Any]],
            aliases: Union[None, Dict[str, str]] = None,
            not_supported: Tuple[str] =(),
            ):
        self.fp_type = fp_type               # The FPSim2 name
        self.fps_typename = fps_typename     # The chemfp name
        self.args = args                     # list of fields which must be in the type string, in order
        self.optional_args = optional_args   # list of fields which are in the type string if not their default value
        if aliases is None:                  # mapping from FPSim2 arg to chemfp arg (chemfp always 'fpSize')
            aliases = {}
        self.aliases = aliases
        self.not_supported = not_supported   # Params which either aren't supported by this driver or by chemfp

    def encode(self, fp_params: Dict[str, Any]) -> str:
        """Convert a dictionary of fingerprint parameters into the chemfp fingerprint type string"""
        terms = [self.fps_typename]
        seen = set()
        for arg in self.args:
            encoder_func = _encoders[arg].encoder
            arg = self.aliases.get(arg, arg)
            value = encoder_func(fp_params[arg])
            seen.add(arg)
            terms.append(f"{arg}={value}")

        for arg, default_value in self.optional_args:
            seen.add(arg)
            value = fp_params[arg]
            if value == default_value:
                continue
            encoder_func = _encoders[arg].encoder
            terms.append(f"{arg}={value}")
            
        for k in fp_params:
            if k not in seen and k not in self.not_supported:
                raise AssertionError(f"not implemented: {k!r}")
        return " ".join(terms)


# Configuration information
_fptype_formatters = (
    # RDKit-Fingerprint/2 minPath=1 maxPath=7 fpSize=2048 nBitsPerHash=2 useHs=1
    #    branchedPaths=1 useBondOrder=1
    FPType(
        "RDKit",
        "RDKit-Fingerprint/2",
        "minPath maxPath fpSize nBitsPerHash useHs".split(),
        (
            ("branchedPaths", True),
            ("useBondOrder", True),
        ),
        not_supported = "fromAtoms minSize tgtDensity atomInvariants atomBits bitInfo".split(),
        ),

    # RDKit-Morgan/1 radius=2 fpSize=2048 useFeatures=0 useChirality=0 useBondTypes=1
    FPType(
        "Morgan",
        "RDKit-Morgan/1",
        "fpSize radius useFeatures useChirality useBondTypes".split(),
        (),
        aliases = {"fpSize": "nBits"},
        not_supported = "fromAtoms invariants".split(),
        ),

    # RDKit-Torsion/2 fpSize=2048 targetSize=4
    FPType(
        "TopologicalTorsion",
        "RDKit-Torsion/2",
        "fpSize targetSize".split(),
        (
            #("nBitsPerEntry", 4),
            ("includeChirality", False),
            ),
        aliases = {"fpSize": "nBits"},
        not_supported = "fromAtoms ignoreAtoms atomInvariants".split(),
        ),

    # RDKit-AtomPair/2 fpSize=2048 minLength=1 maxLength=30
    FPType(
        "AtomPair",
        "RDKit-AtomPair/2",
        "fpSize minLength maxLength".split(),
        (
            ("nBitsPerEntry", 4),
            ("includeChirality", False),
            ("use2D", True),
        ),
        aliases = {"fpSize": "nBits"},
        not_supported = "fromAtoms ignoreAtoms atomInvariants confId".split(),
        ),

    # RDKit-MACCS166/2
    FPType(
        "MACCSKeys",
        "RDKit-MACCS166/2",
        (),
        (),
        ),
        
    # RDKit-Avalon/1 fpSize=512 isQuery=0 bitFlags=15761407
    FPType(
        "Avalon",
        "RDKit-Avalon/1",
        "fpSize isQuery bitFlags".split(),
        (("resetVect", False),),
        aliases = {"fpSize": "nBits"},
        ),
        
    # RDKit-Pattern/4 fpSize=2048
    FPType(
        "RDKPatternFingerprint",
        "RDKit-Pattern/4",
        "fpSize".split(),
        (),
        not_supported = "atomCounts setOnlyBits".split(),
        ),
    )


def get_fps_type_format(fp_type: str, fp_params: Dict[str, Any]):
    """Given the fingerprint type name and parameters, return the chemfp type string"""
    for typeinfo in _fptype_formatters:
        if typeinfo.fp_type == fp_type:
            return typeinfo.encode(fp_params)

    raise KeyError(f"Unknown fingerprint type {fp_type!r}")

## for fp_type, fp_params in FPSim2.io.chem.FP_FUNC_DEFAULTS.items():
##     print("Blah", fp_type, fp_params)
##     print("->", get_fps_type_format(fp_type, fp_params))


#####

parser = argparse.ArgumentParser(
    prog="FPSim2",
    description="FPSim2 driver")

subparsers = parser.add_subparsers()


#####  Create

p = create_parser = subparsers.add_parser(
    "create",
    help="create an FPSim2 data set from a structure file",
    )

g = p.add_mutually_exclusive_group()
      
g.add_argument("--RDKit", action="store_true",
                   help="Generate RDKit hash fingerprints") # FPSim2 name
g.add_argument("--RDK", action="store_true", dest="RDKit",
                   help=argparse.SUPPRESS) # chemfp name

g.add_argument("--Morgan", action="store_true",
                   help="Generate Morgan fingerprints") # FPSim2 name
g.add_argument("--morgan", action="store_true", dest="Morgan",
                   help=argparse.SUPPRESS) # chemfp name

g.add_argument("--TopologicalTorsion", action="store_true",
                   help="Generate Topological Torsion fingerprints") # FPSim2 name
g.add_argument("--torsions", action="store_true", dest="TopologicalTorsion",
                   help=argparse.SUPPRESS) # chemfp name

g.add_argument("--AtomPair", action="store_true",
                   help="Generate Atom Pair fingerprints") # FPSim2 name
g.add_argument("--pairs", action="store_true", dest="AtomPair",
                   help=argparse.SUPPRESS) # chemfp name

g.add_argument("--MACCSKeys", action="store_true",
                   help="Generate 166-bit MACCS keys") # FPSim2 name
g.add_argument("--maccs166", action="store_true", dest="MACCSKeys",
                   help=argparse.SUPPRESS) # chemfp name

g.add_argument("--Avalon", action="store_true",
                   help="Generate Avalon fingerprints") # FPSim2 name
g.add_argument("--avalon", action="store_true", dest="Avalon",
                   help=argparse.SUPPRESS) # chemfp name

g.add_argument("--RDKPatternFingerprint", action="store_true",
                   help="Generate RDKit Pattern (substructure) fingerprints") # FPSim2 name
g.add_argument("--pattern", action="store_true", dest="RDKPatternFingerprint",
                   help=argparse.SUPPRESS) # chemfp name

# Fingerprint parameters


p.add_argument("--fpSize", type=positive_int, metavar="N",
                   help="number of fingerprint bits (Avalon, Morgan, Torsion, AtomPair, RDKit, RDKitPattern)")
p.add_argument("--nBits", type=positive_int, metavar="N",
                   help="alias for --nBits", dest="fpSize")

p.add_argument("--minPath", type=nonnegative_int, metavar="N",
                   help="minimum path length (RDKit)")
p.add_argument("--maxPath", type=nonnegative_int, metavar="N",
                   help="maximum path length (RDKit)")
p.add_argument("--nBitsPerHash", type=nonnegative_int, metavar="N",
                   help="number of bits to set per hash (RDKit)")
p.add_argument("--useHs", type=zero_or_one, metavar="0|1",
                   help="include hydrogens in the fingerprint (RDKit)")
p.add_argument("--useBondOrder", type=zero_or_one, metavar="0|1",
                   help="include bond orders in the fingerprint (RDKit)")
p.add_argument("--branchedPaths", type=zero_or_one, metavar="0|1",
                   help="include branched paths in the fingerprint (RDKit)")

p.add_argument("--radius", type=nonnegative_int, metavar="R",
                   help="Morgan radius (Morgan)")
p.add_argument("--useChirality", type=zero_or_one, metavar="0|1",
                   help="include chirality (Morgan)")
p.add_argument("--useBondTypes", type=zero_or_one, metavar="0|1",
                   help="include bond types (Morgan)")
p.add_argument("--useFeatures", type=zero_or_one, metavar="0|1",
                   help="include features (Morgan)")

p.add_argument("--minLength", type=nonnegative_int, metavar="N",
                   help="minimum length (AtomPair)")
p.add_argument("--maxLength", type=nonnegative_int, metavar="N",
                   help="maximum length (AtomPair)")
p.add_argument("--nBitsPerEntry", type=positive_int, metavar="N",
                   help="number of bits per entry (AtomPair)")
p.add_argument("--includeChirality", type=zero_or_one, metavar="0|1",
                   help="include chirality information (TopologicalTorsion, AtomPair)")

p.add_argument("--use2D", type=zero_or_one, metavar="0|1",
                   help="1 to use 2D, 0 for 3D (requires distances) (AtomPair)")

p.add_argument("--targetSize", type=nonnegative_int, metavar="N",
                   help="target size (TopologicalTorsion)")

p.add_argument("--atomCounts", type=zero_or_one, metavar="0|1",
                   help="include atom counts (RDKPatternFingerprint)")

p.add_argument("--isQuery", type=zero_or_one, metavar="0|1",
                   help="1 to generate a substructure query fingerprint, else 0 (Avalon)")
p.add_argument("--bitFlags", type=nonnegative_int, metavar="N",
                   help="Avalon bit flags (Avalon)")


p.add_argument("-i", "--input", metavar="FILENAME",
                   help="input structure filename")
p.add_argument("--in", metavar="FORMAT", choices=("sdf", "smi"),
                   help="input structure file format (default: use the filename extension)")
p.add_argument("--mol-id-prop",
                   help="SD tag to use for the record identifier")
p.add_argument("--id-tag", dest="id_tag",
                   help="alias for --mol-id-prop")

p.add_argument("--gen-ids", action="store_true",
                   help="generate identifiers")
p.add_argument("--no-sort-by-popcnt", action="store_true",
                   help="do not sort by popcount")

p.add_argument("output", metavar="FILENAME",
                   help="output filename")


def get_fingerprint_type_name(args: argparse.Namespace):
    "Figure out which fingerprint type was selected on the command-line"
    for name in ("RDKit", "Morgan", "TopologicalTorsion", "AtomPair",
                     "MACCSKeys", "Avalon", "RDKPatternFingerprint"):
        if getattr(args, name, False):
            return name
    return "RDKit"

def get_fingerprint_type_name_and_kwargs(args: argparse.Namespace):
    "Figure out which fingerprint type and parameters were selected on the command-line"
    # What fingerprint type name?
    name = get_fingerprint_type_name(args)
    # Which arguments are needed?
    kwargs = FPSim2.io.chem.FP_FUNC_DEFAULTS[name].copy()
    for k in kwargs:
        if k == "nBits":
            # Unalias the special case
            v = getattr(args, "fpSize", None)
        else:
            # Check the command-line arguments
            v = getattr(args, k, None)
        if v is not None:
            # It was specified, so use it. 
            kwargs[k] = v
    
    return name, kwargs


def create_command(parser: argparse.ArgumentParser, args: argparse.Namespace):
    if args.input is None:
        parser.error("Must specify an --input file")

    input_filename = args.input
    try:
        supl = FPSim2.io.chem.get_mol_supplier(input_filename)
        if supl is None:
            raise Exception
    except Exception:
        die("Cannot determine format (only uncompressed '.smi' and '.sdf' files are supported)")
    
    fp_typename, fp_kwargs = get_fingerprint_type_name_and_kwargs(args)
    
    FPSim2.io.create_db_file(input_filename, args.output, fp_typename, fp_kwargs,
                                 mol_id_prop = args.mol_id_prop,
                                 gen_ids = args.gen_ids,
                                 sort_by_popcnt = not args.no_sort_by_popcnt)

p.set_defaults(command=create_command,
                   subparser=p)

#####  Import

p = import_parser = subparsers.add_parser(
    "import",
    help="import RDKit fingerprints in FPS format",
    )

p.add_argument("--input", "-i", metavar="FILENAME",
                   help="input fingerprints")
p.add_argument("--gen-ids", action="store_true",
                   help="generate identifiers")
p.add_argument("--remove-prefix", metavar="PREFIX",
                   help="remove PREFIX before converting the id to an int")
p.add_argument("--no-sort-by-popcnt", action="store_true",
                   help="do not sort by popcount")

p.add_argument("output", metavar="FILENAME",
                   help="output filename")

def parse_chemfp_type(chemfp_type: str) -> Tuple[str, Dict[str, Any]]:
    """Convert a chemfp fingerprint type string into its FPSim2 fingerprint name and kwargs

    For example:
      >>> parse_chemfp_type("RDKit-Morgan/1 nBits=2048 radius=2 useFeatures=0 useChirality=0 useBondTypes=1")
      ('Morgan', {'radius': 2, 'nBits': 2048, 'invariants': [], 'fromAtoms': [],
           'useChirality': 0, 'useBondTypes': 1, 'useFeatures': 0})

    """
    terms = chemfp_type.split()

    fps_typename = terms[0]
    if not fps_typename.startswith("RDKit-"):
        raise ValueError(f"FPSim2 only supports RDKit fingerprints: {fps_typename}")

    # Ignore versions when looking for a match
    prefix = fps_typename.partition("/")[0] + "/"
    for formatter in _fptype_formatters:
        if formatter.fps_typename.startswith(prefix):
            break
    else:
        raise ValueError(f"Unsupported fingerprint type: {fps_typename!r}")

    kwargs = FPSim2.io.chem.FP_FUNC_DEFAULTS[formatter.fp_type].copy()

    # Parse the parameters into a dictionary for easy lookup (and check for duplicates)
    term_dict = dict()
    for term in terms[1:]:
        left, mid, right = term.partition("=")
        if mid != "=":
            raise ValueError(f"Cannot parse term {term!r} in {chemfp_type!r}")
        if left in term_dict:
            raise ValueError(f"Duplicate term {term!r} in {chemfp_type!r}")
        term_dict[left] = right

    aliases = formatter.aliases

    possible_args = list(formatter.args) + [arg for (arg, default_value) in formatter.optional_args]

    for arg in possible_args:
        if arg not in term_dict:
            continue
        value_str = term_dict.pop(arg)
        try:
            value = _encoders[arg](value_str)
        except argparse.ArgumentTypeError as err:
            raise ValueError(f"{arg} {err}: {value_str!r}")
        kwargs[aliases.get(arg, arg)] = value
    
    if term_dict:
        keys = sorted(term_dict)
        if len(keys) > 1:
            die(f"Unsupported parameters: {keys}")
        else:
            die(f"Unsupported parameter: {keys[0]}")
    return formatter.fp_type, kwargs
        
def check_chemfp_num_bits(num_bits: int, fp_type: str, fp_params: Dict[str, Any]) -> int:
    "check that num_bits (if not None) matches the fingerprint type and parameters; return the number of bits"
    expected_num_bits = FPSim2.io.chem.get_fp_length(fp_type, fp_params)
    if num_bits is None:
        return expected_num_bits
    if num_bits != expected_num_bits:
        die(f"FPS file header 'num_bits' is {num_bits} but the type says it should be {expected_num_bits}")
    return num_bits


def open_fps_file(infile: Iterable[str]) -> Tuple[str, Dict[str, Any], str, Iterable[Tuple[str, List[int]]]]:
    """open an FPS file, read the header, and get ready to read the fingerprints

    return (fp_type, fp_params, rdki
fp_type, fp_params, rdkit_version, parse_chemfp_maccs_fingerprints(fingerprint_reader)
    """
    it = iter(infile)
    num_bits = None
    chemfp_type = None
    fp_type = fp_params = None
    seen_rdkit_version = False
    rdkit_version = "unknown"

    # Read the header
    for lineno, line in enumerate(it, 1):
        # No longer in the header
        if line[:1] != "#":
            break
        
        if line.startswith("#num_bits="):
            if num_bits is not None:
                die("FPS file contains multiple 'num_bits' headers")
            s = line[10:].strip()
            if not s.isdigit():
                die("FPS file 'num_bits' header value must be an integer")
            num_bits = int(s)
            
        elif line.startswith("#type="):
            if chemfp_type is not None:
                die("FPS file contains multiple 'type' headers")
            chemfp_type = line[6:].strip()
            try:
                fp_type, fp_params = parse_chemfp_type(chemfp_type)
            except ValueError as err:
                die(f"Cannot parse 'type' header at line {lineno} of FPS file: {err}")
        
        elif line.startswith("#software="):
            if seen_rdkit_version:
                die("FPS file contains multiple 'software' headers")
            seen_rdkit_version = True
            terms = line[10:].split()
            for term in terms:
                if term.startswith("RDKit/"):
                    rdkit_version = term[6:]
                    break
            # else: use the default of 'unknown'
    else:
        # No fingerprints
        if chemfp_type is None:
            die("no fingerprint type available")
        num_bits = check_chemfp_num_bits(num_bits, fp_type, fp_params)
        return fp_type, fp_params, rdkit_version, ()

    # Have at least one fingerprint
    if chemfp_type is None:
        die("no fingerprint type available")
    num_bits = check_chemfp_num_bits(num_bits, fp_type, fp_params)

    # 'line' contains the first fingerprint line, 'it' contains the input iterator
    fingerprint_reader = read_fingerprints(lineno, num_bits, line, it)
    
    if fp_type == "MACCSKeys":
        if num_bits != 166:
            die("MACCSKeys may only have 166 bits")
        
        return fp_type, fp_params, rdkit_version, parse_chemfp_maccs_fingerprints(fingerprint_reader)

    if num_bits % 64 == 0:
        # use a converter which is more optimized for well-aligned data
        return fp_type, fp_params, rdkit_version, parse_aligned_chemfp_fingerprints(num_bits, fingerprint_reader)
    else:
        return fp_type, fp_params, rdkit_version, parse_unaligned_chemfp_fingerprints(num_bits, fingerprint_reader)

def read_fingerprints(lineno: int, num_bits: int, first_line: str, it: Iterable[str]) -> Iterable[Tuple[str, int]]:
    "iterate through the fingerprint data of an FPS file"
    expected_num_hex_digits = ((num_bits+7)//8) * 2

    # Process the first line
    columns = first_line.rstrip("\n").split("\t")
    if len(columns) < 2:
        die(f"FPS file line {lineno} must have a hex-encoded fingerprint and identifier")
    fp_str = columns[0]
    if len(fp_str) != expected_num_hex_digits:
        die(f"FPS file line {lineno} has a fingerprint of length {len(fp_str)} while the fingerprint should have {num_bits} bits")
    # Convert to bytes, reverse the bit order and the byte order, then convert to an integer as little-endian bytes
    yield columns[1], int.from_bytes(bytes.fromhex(fp_str).translate(_reverse_bits_transtable)[::-1], "little")
    
    del first_line # safety check to prevent re-use in the following copy&paste code

    # Process the remaining lines
    for lineno, line in enumerate(it, lineno+1):
        columns = line.rstrip("\n").split("\t")
        if len(columns) < 2:
            die(f"FPS file line {lineno} must have a hex-encoded fingerprint and identifier")
        fp_str = columns[0]
        if len(fp_str) != expected_num_hex_digits:
            die(f"FPS file line {lineno} has a fingerprint of length {len(fp_str)} while the fingerprint should have {num_bits} bits")
        yield columns[1], int.from_bytes(bytes.fromhex(fp_str).translate(_reverse_bits_transtable)[::-1], "little")

def parse_chemfp_maccs_fingerprints(fingerprint_reader: Iterable[Tuple[str, int]]) -> Iterable[Tuple[str, List[int]]]:
    # For the MACCS fingerprints, which need a shift to FPSim2 standard form.
    # Chemfp has key 1 in bit 0. FPSim2 has key 1 in bit 1
    for id, int_fp in fingerprint_reader:
        yield id, [
            ((int_fp >> 105) & 0xffffffffffffffff),
            ((int_fp >> 41) & 0xffffffffffffffff),
            ((int_fp >>  2) & 0x7fffffffff),
            ]
        
def parse_aligned_chemfp_fingerprints(num_bits: int,
                                      fingerprint_reader: Iterable[Tuple[str, int]]) \
                                      -> Iterable[Tuple[str, List[int]]]:
    # This function is for fingerprints lengths which are a multiple of 64-bit
    assert num_bits % 64 == 0
    num_fp_fields = num_bits//64
    shifts = [64*(num_fp_fields-i-1) for i in range(num_fp_fields)]
    
    for id, int_fp in fingerprint_reader:
        yield id, [((int_fp >> shift) & 0xffffffffffffffff) for shift in shifts]
            
def parse_unaligned_chemfp_fingerprints(num_bits, fingerprint_reader):
    # This function is for fingerprints lengths which are NOT a multiple of 64-bit.
    # It requires extra work to shift and mask the words correctly, especially the last.
    num_fp_fields = (num_bits + 63)//64
    num_bytes = (num_bits+7) // 8
    delta_bits = (num_fp_fields * 64 - num_bytes * 8)
    shifts_and_masks = [(64*(num_fp_fields-i-1) - delta_bits, 0xffffffffffffffff) for i in range(num_fp_fields)]

    # The bytes in the last word needs to be right-aligned, not left-aligned.
    num_final_bits = 8 - num_bits % 8
    shifts_and_masks[-1] = (num_final_bits, (1<<(64-delta_bits))-1)
    
    for id, int_fp in fingerprint_reader:
        yield id, [((int_fp >> shift) & mask) for (shift, mask) in shifts_and_masks]

def import_command(parser: argparse.ArgumentParser, args: argparse.Namespace):
    if args.input is None:
        infile = sys.stdin
    else:
        infile = open(args.input)

    fp_typename, fp_params, rdkit_version, fingerprint_reader = open_fps_file(infile)
    if args.gen_ids:
        # Generate a new sequential identifier for each record, starting from 1
        fingerprint_reader = ((mol_id, fp) for (mol_id, (id, fp)) in enumerate(fingerprint_reader, 1))
    else:
        if args.remove_prefix is None:
            # Convert directly to an integer, or give a helpful exception if there's a problem
            def _int(s: str):
                try:
                    return int(s)
                except ValueError:
                    raise Exception(
                        "FPSim only supports integer ids for molecules, "
                        "consider setting --gen-ids or using --remove-prefix to remove an identifier prefix"
                        )
        else:
            # Remove a leading prefix, like "CHEMBL", and keep the remaining sequence number
            prefix = args.remove_prefix
            prefix_len = len(prefix)
            def _int(s: str):
                if s[:prefix_len] == prefix:
                    try:
                        return int(s[prefix_len:])
                    except ValueError:
                        raise Exception(f"identifier {s!r} is not an integer after removing the prefix {prefix!r}")
                else:
                    raise Exception(f"identifier {s!r} does not start with prefix {prefix!r}")
        # Normalize the identifiers to an integer
        fingerprint_reader = ((_int(id), fp) for (id, fp) in fingerprint_reader)
        

    # create_db_file() has no way to pass in the fingerprints to use.  Rather
    # than re-implement the underlying functionality, it's easier to replace
    # the existing get_mol_supplier() and rdmol_to_efp() so they pass
    # fingerprints through unchanged.
    def import_get_mol_supplier(mols_source):
        def open_existing_fingerprint_reader(mols_source, gen_ids, mol_id_prop):
            return fingerprint_reader
        return open_existing_fingerprint_reader

    def import_rdmol_to_efp(rdmol, fp_type, fp_params):
        return rdmol  # 'rdmol' is really the fingerprint
    
    from FPSim2.io.backends import pytables
    pytables.get_mol_supplier = import_get_mol_supplier
    pytables.rdmol_to_efp = import_rdmol_to_efp

    unused = None
    FPSim2.io.create_db_file(unused, args.output, fp_typename, fp_params,
                                 mol_id_prop = unused,
                                 gen_ids = unused,
                                 sort_by_popcnt = not args.no_sort_by_popcnt)
    

p.set_defaults(command=import_command,
                   subparser=p)

#####  Export

p = export_parser = subparsers.add_parser(
    "export",
    help="export the fingerprints to FPS format",
    )

p.add_argument("--output", "-o", metavar="FILENAME",
                   help="where to write the FPS data")

p.add_argument("input", metavar="FILENAME",
                   help="FPSim2 PyTables file")

_reverse_bits_transtable = (
    b'\x00\x80@\xc0 \xa0`\xe0\x10\x90P\xd00\xb0p\xf0\x08\x88H\xc8(\xa8h\xe8\x18\x98X\xd88\xb8x\xf8'
    b'\x04\x84D\xc4$\xa4d\xe4\x14\x94T\xd44\xb4t\xf4\x0c\x8cL\xcc,\xacl\xec\x1c\x9c\\\xdc<\xbc|\xfc'
    b'\x02\x82B\xc2"\xa2b\xe2\x12\x92R\xd22\xb2r\xf2\n\x8aJ\xca*\xaaj\xea\x1a\x9aZ\xda:\xbaz\xfa'
    b'\x06\x86F\xc6&\xa6f\xe6\x16\x96V\xd66\xb6v\xf6\x0e\x8eN\xce.\xaen\xee\x1e\x9e^\xde>\xbe~\xfe'
    b'\x01\x81A\xc1!\xa1a\xe1\x11\x91Q\xd11\xb1q\xf1\t\x89I\xc9)\xa9i\xe9\x19\x99Y\xd99\xb9y\xf9'
    b'\x05\x85E\xc5%\xa5e\xe5\x15\x95U\xd55\xb5u\xf5\r\x8dM\xcd-\xadm\xed\x1d\x9d]\xdd=\xbd}\xfd'
    b'\x03\x83C\xc3#\xa3c\xe3\x13\x93S\xd33\xb3s\xf3\x0b\x8bK\xcb+\xabk\xeb\x1b\x9b[\xdb;\xbb{\xfb'
    b"\x07\x87G\xc7'\xa7g\xe7\x17\x97W\xd77\xb7w\xf7\x0f\x8fO\xcf/\xafo\xef\x1f\x9f_\xdf?\xbf\x7f\xff"
    )

def export_command(parser: argparse.ArgumentParser, args: argparse.Namespace):
    from FPSim2.io.backends import pytables
    
    back_end = pytables.PyTablesStorageBackend(
        args.input,
        in_memory_fps=False)

    # Figure out the back-end fingerprint information, and the mapping to chemfp
    fp_type, fp_params, rdkit_ver = back_end.read_parameters()
    fps_typename = get_fps_type_format(fp_type, fp_params)
    num_bits = FPSim2.io.chem.get_fp_length(fp_type, fp_params)

    # Open the output file
    if args.output is None:
        outfile = sys.stdout
    else:
        outfile = open(args.output, "w")
    write = outfile.write

    # Write the header
    write("#FPS1\n")
    write(f"#num_bits={num_bits}\n")
    write(f"#type={fps_typename}\n")
    write(f"#software=RDKit/{rdkit_ver} FPSim2/{FPSim2.__version__}\n")
    
    # Table is laid out as:
    #   "fp_id": Int64Col(shape=(), dflt=0, pos=0),
    #   "f1": UInt64Col(shape=(), dflt=0, pos=1),
    #   "f2": UInt64Col(shape=(), dflt=0, pos=2),
    #     ....
    #   "f31": UInt64Col(shape=(), dflt=0, pos=31),
    #   "f32": UInt64Col(shape=(), dflt=0, pos=32),
    #   "popcnt": Int64Col(shape=(), dflt=0, pos=33)}
    with tb.open_file(args.input, mode="r") as fp_file:
        fps = fp_file.root.fps
        total_num_rows = len(fps)
        if total_num_rows == 0:
            return # Finished

        num_fields = len(fps[0])
        assert num_fields > 2, num_fields
        
        if fp_type == "MACCSKeys":
            # Special treatment for the MACCS Keys to align the bit numbers
            if num_bits != 166:
                raise AssertionError(f"MACCSKeys length is not supposed to be {num_bits}")
            dtype = [("fp_id", "<i8"), ("fps", "<u8", 3), ("popcnt", "<u8")]
            for i in range(0, total_num_rows, 10000):
                arr = fps.read(start=i, stop=i+10000)
                data = arr.view(dtype=dtype)
                for fp_id, (f1, f2, f3), popcnt in data:
                    # RDKit stores key 1 as bit 1, so bit 0 is always 0.
                    # chemfp stores key 1 as bit 0
                    # This requires a bit-shift.
                    # FPSim2 stores RDKit bits in little-endian order:
                    #   bits 0-63 in the first word; (bit 0 is key 64, bit 62 is key 1)
                    #   bits 64-127 in the second word;
                    #   bits 128-167 right-shifted 25 bits into the third word. (bit 0 is key 128)
                    bits = ((int(f1) << 129) + (int(f2)<<65) + (int(f3)<<26))
                    # FPSim2 
                    b = bits.to_bytes(32, "big")
                    b = b.translate(_reverse_bits_transtable)
                    hex_b = b.hex()
                    write(f"{hex_b[16:58]}\t{fp_id}\n")
            
        else:
            num_fp_fields = num_fields - 2
            dtype = [("fp_id", "<i8"), ("fps", "<u8", num_fp_fields), ("popcnt", "<u8")]
            join_indices = [(i, i+8) for i in range(num_fp_fields*8-8, -1, -8)]

            if num_bits % 64 == 0:
                # 64-bit aligned is easier to convert to the FPS byte order
                for i in range(0, total_num_rows, 10000):
                    arr = fps.read(start=i, stop=i+10000)
                    data = arr.view(dtype=dtype)
                    for fp_id, fps_row, popcnt in data:
                        fps_bytes = fps_row.tobytes().translate(_reverse_bits_transtable)[::-1]
                        b = b"".join(fps_bytes[i:j] for i,j in join_indices)
                        write(f"{b.hex()}\t{fp_id}\n")
            else:
                # non-64-bit aligned is harder because the last word is right-aligned
                remainder = num_bits % 64
                shift_bits = np.uint8(8-remainder%8)
                join_indices = [(i, i+8) for i in range(num_fp_fields*8-8, -1, -8)]
                join_indices[-1] = (8-((remainder+7)//8), 8)
                
                for i in range(0, total_num_rows, 10000):
                    arr = fps.read(start=i, stop=i+10000)
                    data = arr.view(dtype=dtype)
                    for fp_id, fps_row, popcnt in data:
                        fps_row = fps_row.copy()
                        fps_row[-1] <<= shift_bits
                        fps_bytes = fps_row.tobytes().translate(_reverse_bits_transtable)[::-1]
                        b = b"".join(fps_bytes[i:j] for i,j in join_indices)
                        write(f"{b.hex()}\t{fp_id}\n")


p.set_defaults(command=export_command,
                   subparser=p)


##### Set common grouped arguments

def add_query_arguments(p: argparse.ArgumentParser):
    p.add_argument("--queries", metavar="FILENAME",
                       help="query structures, in an uncompressed SDF or SMILES files")
    p.add_argument("--query", metavar="STRING",
                       help="a query SMILES string")

def add_structure_file_arguments(p: argparse.ArgumentParser):
    p.add_argument("--mol-id-prop",
                       help="SD tag to use for the record identifier")
    p.add_argument("--id-tag", dest="id_tag",
                       help="alias for --mol-id-prop")

    p.add_argument("--has-header", action="store_true",
                       help="the first line of the SMILES file is a header")
    p.add_argument("--delimiter", choices=("tab", "space", "comma"),
                       help="delimiter (one of 'tab', 'space', or 'comma'); default: space")


    
#####  Tanimoto and Tverksy search

p = search_parser = subparsers.add_parser(
    "search",
    help="Tanimoto or Tversky search",
    )

add_query_arguments(p)

p.add_argument("-t", "--threshold", metavar="FLOAT", type=nonnegative_float, default=0.7,
                   help="minimum similarity threshold")
p.add_argument("-a", "--alpha", type=nonnegative_float,
                   help="Tversky alpha value")
p.add_argument("-b", "--beta", type=nonnegative_float,
                   help="Tversky beta value")

add_structure_file_arguments(p)

p.add_argument("--on-disk", action="store_true",
                   help="do an on-disk search instead of in-memory")
p.add_argument("-j", "--n-workers", type=positive_int, metavar="N", default=1)
p.add_argument("-o", "--output", metavar="FILENAME",
                   help="write the results to FILENAME instead of stdout")
p.add_argument("--times", action="store_true",
                   help="display times for different stages to stderr")
p.add_argument("targets",
                   help="FPSim2 PyTables file")

def iter_with_prop_name(suppl, prop_name):
    for recno, mol in enumerate(suppl, 1):
        if mol is None:
            continue
        try:
            yield mol.GetProp(prop_name), mol
        except KeyError:
            yield f"Query{recno}", mol


def get_reader(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    from rdkit import Chem
    
    if args.query is None:
        if args.queries is None:
            parser.error("Must specify one of --query or --queries")


        ## Specified a structure file
        queries = args.queries

        if queries.endswith(".sdf"):
            suppl = Chem.ForwardSDMolSupplier(queries)
            if args.mol_id_tag is None:
                return iter_with_prop_name(suppl, "_Name")
            else:
                return iter_with_prop_name(suppl, args.mol_id_tag)
            
        if queries.endswith(".smi"):
            delimiter = {
                None: " ",    # default
                "tab": "\t",
                "space": " ",
                "comma": ",",
                }[args.delimiter]
            suppl = Chem.SmilesMolSupplier(queries, delimiter=delimiter, titleLine=not args.has_header)
            return iter_with_prop_name(suppl, "_Name")

        die("Cannot determine format (only uncompressed '.smi' and '.sdf' files are supported)")
        
    else:
        # Specified a SMILES on the command-line
        mol = Chem.MolFromSmiles(args.query)
        if mol is None:
            die(f"Cannot parse --query {args.query!r} as SMILES")

        # helper function to fit the expected iterable API
        def read_commandline_smiles() -> Iterable[Tuple[str, Chem.Mol]]:
            yield "Query", mol
        return read_commandline_smiles()

# The following subclass records the load_query() time.
# Normally this is the time to parse the input structure ("load_molecule()")
# and generate a fingerprint query ("rdmol_to_efp()").
# 
# The search_command() parses the molecule itself, so it replaces
# load_molecule() with a no-op function. This means load_query()
# ends up timing just the fingerprint generation time.

class InstrumentedFPSim2Engine(FPSim2.FPSim2Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fpgen_time = 0.0

    def load_query(self, query_string: str) -> np.ndarray:
        t1 = time.time()
        result = super().load_query(query_string)
        t2 = time.time()
        self.fpgen_time += (t2-t1)
        return result

def format_filename(s: str) -> str:
    "remove characters which aren't supported in the the report file"
    return s.replace("\n", "").replace("\0", "").replace("\r", "")
    
def search_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    
    # Check the Tversky parameters, if given
    alpha = args.alpha
    beta = args.beta
    if alpha is None:
        if beta is None:
            pass
        else:
            parser.error("Cannot specify --beta without also specifying --alpha")
    else:
        if alpha < 0.0:
            die("--alpha must be non-negative")
            
        if beta is None:
            if alpha == 0.0:
                die("one of --alpha and --beta must be positive")
            beta = alpha
        else:
            if beta < 0.0:
                die("--beta must be non-negative")

    threshold = args.threshold
    n_workers = args.n_workers

    # Prepare to read structures
    reader = get_reader(parser, args)

    # This reader reads (id, rdmol) pairs but the underlying
    # code expects a query string as input, not a molecule.
    # Intercept the FPSim2/base.py so it accepts a molecule
    # as input rather than a string. (Note: the type signatures
    # are then wrong.)
    from FPSim2 import base
    def load_molecule(mol_string: Any) -> Any:
        return mol_string
    base.load_molecule = load_molecule

    # Open the output file
    if args.output:
        outfile = open(args.output, "w")
        write = outfile.write
    else:
        write = sys.stdout.write

    # Figure out which search engine to use
    report_times = args.times
    if report_times:
        engine_class = InstrumentedFPSim2Engine
    else:
        engine_class = FPSim2.FPSim2Engine    

    # Load the target data
    get_time = time.time
    start_load_time = get_time()
    engine = engine_class(
        fp_filename = args.targets,
        in_memory_fps = not args.on_disk)
    end_load_time = get_time()

    # Write the header
    write("#Simsearch/1\n")

    num_bits = FPSim2.io.chem.get_fp_length(engine.fp_type, engine.fp_params)
    write(f"#num_bits={num_bits}\n")
    if alpha is None:
        write(f"#type=Tanimoto k=all threshold={threshold}\n")
    else:
        write(f"#type=Tversky k=all threshold={threshold} alpha={alpha} beta={beta}\n")
    write(f"#software=FPSim2/{FPSim2.__version__}\n")
    if args.queries is not None:
        write(f"#queries={format_filename(args.queries)}\n")
    write(f"#targets={format_filename(args.targets)}\n")
    
    # Start search time tracking
    start_time = get_time()

    total_output_time = 0.0
    total_search_time = 0.0
    total_parse_time = 0.0

    # Dispatch to the correct search
    try:
        if args.on_disk:
            if alpha is None:
                while 1:
                    # This while-loop is normally written as a for-loop,
                    #   for id, mol in reader:
                    # Breaking it out as a while-loop makes it easier
                    # to time the structure reader.
                    t0 = get_time()
                    id, mol = next(reader)
                    t1 = get_time()
                    result = engine.on_disk_similarity(mol, threshold, n_workers)
                    t2 = get_time()
                    formatted_hits = "\t".join(f"{hit_id}\t{hit_score}" for (hit_id, hit_score) in result)
                    write(f"{len(result)}\t{id}\t{formatted_hits}\n")
                    total_output_time += get_time() - t2
                    total_search_time += t2-t1
                    total_parse_time += t1-t0
            else:
                while 1:
                    t0 = get_time()
                    id, mol = next(reader)
                    t1 = get_time()
                    result = engine.on_disk_tversky(mol, threshold, alpha, beta, n_workers)
                    t2 = get_time()
                    formatted_hits = "\t".join(f"{hit_id}\t{hit_score}" for (hit_id, hit_score) in result)
                    write(f"{len(result)}\t{id}\t{formatted_hits}\n")
                    total_output_time += get_time() - t2
                    total_search_time += t2-t1
                    total_parse_time += t1-t0

        else:
            if alpha is None:
                while 1:
                    t0 = get_time()
                    id, mol = next(reader)
                    t1 = get_time()
                    result = engine.similarity(mol, threshold, n_workers)
                    t2 = get_time()
                    formatted_hits = "\t".join(f"{hit_id}\t{hit_score}" for (hit_id, hit_score) in result)
                    write(f"{len(result)}\t{id}\t{formatted_hits}\n")
                    total_output_time += get_time() - t2
                    total_search_time += t2-t1
                    total_parse_time += t1-t0
            else:
                while 1:
                    t0 = get_time()
                    id, mol = next(reader)
                    t1 = get_time()
                    result = engine.tversky(mol, threshold, alpha, beta, n_workers)
                    t2 = get_time()
                    formatted_hits = "\t".join(f"{hit_id}\t{hit_score}" for (hit_id, hit_score) in result)
                    write(f"{len(result)}\t{id}\t{formatted_hits}\n")
                    total_output_time += get_time() - t2
                    total_search_time += t2-t1
                    total_parse_time += t1-t0
    except StopIteration:
        # handle the end-of-iteration for the structure readers
        pass

    end_time = get_time()
    
    if report_times:
        load_time = end_load_time - start_load_time
        total_time = end_time - start_time
        fpgen_time = engine.fpgen_time
        total_search_time -= fpgen_time
        
        sys.stderr.write(f"load {load_time:.2f} parse {total_parse_time:.2f} fp-gen: {fpgen_time:.2f} search {total_search_time:.2f} output {total_output_time:.2f} total {total_time + load_time:.2f}\n")
    

p.set_defaults(command=search_command,
                   subparser=p)

#####  substructure screening

p = substructure_parser = subparsers.add_parser(
    "substructure",
    help="substructure screen",
    )

add_query_arguments(p)

add_structure_file_arguments(p)

p.add_argument("--on-disk", action="store_true",
                   help="do an on-disk search instead of in-memory")
p.add_argument("-j", "--n-workers", type=positive_int, metavar="N", default=1)
p.add_argument("-o", "--output", metavar="FILENAME",
                   help="write the results to FILENAME instead of stdout")
p.add_argument("--times", action="store_true",
                   help="display times for different stages to stderr")
p.add_argument("targets",
                   help="FPSim2 PyTables file")


def substructure_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    n_workers = args.n_workers
    
    reader = get_reader(parser, args)

    # This reader reads (id, rdmol) pairs but the underlying
    # code expects a query string as input, not a molecule.
    # Intercept the FPSim2/base.py so it accepts a molecule
    # as input rather than a string. (Note: the type signatures
    # are then wrong.)
    from FPSim2 import base
    def load_molecule(mol_string: Any) -> Any:
        return mol_string
    base.load_molecule = load_molecule

    # Open the output file
    if args.output:
        outfile = open(args.output, "w")
        write = outfile.write
    else:
        write = sys.stdout.write

    # Figure out which search engine to use
    report_times = args.times
    if report_times:
        engine_class = InstrumentedFPSim2Engine
    else:
        engine_class = FPSim2.FPSim2Engine    

    get_time = time.time
    start_load_time = get_time()
    engine = engine_class(
        fp_filename = args.targets,
        in_memory_fps = not args.on_disk)
    end_load_time = get_time()

    # Write the header
    write("#Screenout/1\n")
    write(f"#software=FPSim2/{FPSim2.__version__}\n")
    if args.queries is not None:
        write(f"#queries={format_filename(args.queries)}\n")
    write(f"#targets={format_filename(args.targets)}\n")
    
    # Start screenout time tracking
    start_time = get_time()

    total_output_time = 0.0
    total_screenout_time = 0.0
    total_parse_time = 0.0

    # Dispatch to the correct search
    try:
        if args.on_disk:
            while 1:
                t0 = get_time()
                id, mol = next(reader)
                t1 = get_time()
                result = engine.on_disk_substructure(mol, n_workers)
                t2 = get_time()
                formatted_hits = "\t".join(f"{hit_id}\t{hit_score}" for (hit_id, hit_score) in result)
                write(f"{len(result)}\t{id}\t{formatted_hits}\n")
                total_output_time += get_time() - t2
                total_screenout_time += t2-t1
                total_parse_time += t1-t0
        else:
            while 1:
                t0 = get_time()
                id, mol = next(reader)
                t1 = get_time()
                result = engine.substructure(mol, n_workers)
                t2 = get_time()
                formatted_hits = "\t".join(map(str, result))
                write(f"{len(result)}\t{id}\t{formatted_hits}\n")
                total_output_time += get_time() - t2
                total_screenout_time += t2-t1
                total_parse_time += t1-t0
    except StopIteration:
        # Use an explicit while-loop instead of a for-loop to
        # make it easier to get timings for the structure reader.
        # See the comments in 'search_command()' for more info.
        pass

    end_time = get_time()
    
    if report_times:
        load_time = end_load_time - start_load_time
        total_time = end_time - start_time
        fpgen_time = engine.fpgen_time
        total_screenout_time -= fpgen_time
        
        sys.stderr.write(f"load {load_time:.2f} parse {total_parse_time:.2f} fp-gen: {fpgen_time:.2f} screen {total_screenout_time:.2f} output {total_output_time:.2f} total {total_time + load_time:.2f}\n")


p.set_defaults(command=substructure_command,
                   subparser=p)

########### symmetric distance matrix

p = distmat_parser = subparsers.add_parser(
    "distmat",
    help="generate symmetric distance matrix and save in SciPy's sparse 'npz' format",
    )

p.add_argument("-t", "--threshold", metavar="FLOAT", type=nonnegative_float, default=0.7,
                   help="minimum similarity threshold")
p.add_argument("-a", "--alpha", type=nonnegative_float,
                   help="Tversky alpha and beta value (must be equal to be symmetric)")

p.add_argument("-j", "--n-workers", type=positive_int, metavar="N", default=1,
                   help="number of worker threads (default: 1)")
p.add_argument("-o", "--output", metavar="FILENAME",
                   help="write the results to FILENAME instead of stdout")
p.add_argument("--times", action="store_true",
                   help="display times for different stages to stderr")
p.add_argument("targets",
                   help="FPSim2 PyTables file")


def distmat_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    import scipy.sparse

    # Configure the search
    threshold = args.threshold
    n_workers = args.n_workers

    if args.alpha is None:
        search_type = "tanimoto"
        alpha = beta = 0
    else:
        search_type = "tversky"
        alpha = beta = args.alpha

    # Load the data set into memory and record the time.
    start_load_time = time.time()
    engine = FPSim2.FPSim2Engine(
        fp_filename = args.targets,
        in_memory_fps = True,
        )
    load_time = time.time() - start_load_time

    # Prepare the output
    if args.output is None:
        outfile = sys.stdout.buffer
        if outfile.isatty():
            # Put a warning now so people won't be annoyed an hour later when the results are
            # dumped to stdout instead of to a pipe or file.
            sys.stderr.write(
                "WARNING: the compressed matrix will be written to stdout, which appears to be a terminal.\n")
    else:
        outfile = open(args.output, "wb")

    # Do the search and save the results
    t1 = time.time()
    csr_matrix = engine.symmetric_distance_matrix(threshold, search_type, alpha, beta, n_workers)
    t2 = time.time()
    scipy.sparse.save_npz(outfile, csr_matrix)
    t3 = time.time()

    # Report times
    if args.times:
        sys.stderr.write(f"load {load_time:.2f} search {t2-t1:.2f} output: {t3-t2:.2f} total {t3-t1 + load_time:.2f}\n")

    
p.set_defaults(command=distmat_command,
                   subparser=p)

###########

def main(args: Union[None, Dict[str, str]] = None) -> None:
    args = parser.parse_args(args)
    command = getattr(args, "command", None)
    if command is None:
        parser.error("missing subcommand")
    args.command(args.subparser, args)
    
if __name__ == "__main__":
    main()
