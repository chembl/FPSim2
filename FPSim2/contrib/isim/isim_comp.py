from .utils import unpack_fpe, get_c_total_from_fpe
import numpy as np

"""                             iSIM_MODULES
    ----------------------------------------------------------------------
    
    Miranda-Quintana Group, Department of Chemistry, University of Florida 
    
    ----------------------------------------------------------------------
    
    Please, cite the original paper on iSIM:
    https://doi.org/10.26434/chemrxiv-2023-fxlxg
    """


def _calculate_counters(data, n_objects, k=1):
    """Calculate 1-similarity, 0-similarity, and dissimilarity counters

    Arguments
    ---------
    data : np.ndarray
        Array of arrays, each sub-array contains the binary object
        OR Array with the columnwise sum, if so specify n_objects

    n_objects : int
        Number of objects.

    k : int
        Integer indicating the 1/k power used to approximate the average of the
        similarity values elevated to 1/k.

    Returns
    -------
    counters : dict
        Dictionary with the weighted and non-weighted counters.

    """
    if data.ndim == 1:
        c_total = data
    else:
        c_total = np.sum(data, axis=0)

    # Calculate a, d, b + c
    a_array = c_total * (c_total - 1) / 2
    off_coincidences = n_objects - c_total
    d_array = off_coincidences * (off_coincidences - 1) / 2
    dis_array = off_coincidences * c_total

    a = np.sum(np.power(a_array, 1 / k))
    d = np.sum(np.power(d_array, 1 / k))
    total_dis = np.sum(np.power(dis_array, 1 / k))

    total_sim = a + d
    p = total_sim + total_dis

    counters = {"a": a, "d": d, "total_sim": total_sim, "total_dis": total_dis, "p": p}
    return counters


def get_sim_dict(fpe, k=1):
    """Calculate a dictionary containing all the available similarity indexes

    Arguments
    ---------
    fpe : FPSim2.FPSim2Engine
        FPSim2 engine

    k : int
        Integer indicating the 1/k power used to approximate the average of the
        similarity values elevated to 1/k.

    Returns
    -------
    sim_dict : dict
        Dictionary with the weighted and non-weighted similarity indexes."""
    data = get_c_total_from_fpe(fpe)
    n_objects = fpe.fps.shape[0]
    return _gen_sim_dict(data, n_objects, k=k)


def _gen_sim_dict(data, n_objects, k=1):
    """Calculate a dictionary containing all the available similarity indexes

    Arguments
    ---------
    data : np.ndarray
        Array of arrays, each sub-array contains the binary object
        OR Array with the columnwise sum, if so specify n_objects

    n_objects : int
        Number of objects.

    k : int
        Integer indicating the 1/k power used to approximate the average of the
        similarity values elevated to 1/k.

    Returns
    -------
    sim_dict : dict
        Dictionary with the weighted and non-weighted similarity indexes."""

    # Calculate the similarity and dissimilarity counters
    counters = _calculate_counters(data=data, n_objects=n_objects, k=k)
    jt = (counters["a"]) / (counters["a"] + counters["total_dis"])
    rr = (counters["a"]) / (counters["p"])
    sm = (counters["total_sim"]) / (counters["p"])

    # Dictionary with all the results
    Indices = {
        "JT": jt,  # JT: Jaccard-Tanimoto
        "RR": rr,  # RR: Russel-Rao
        "SM": sm,  # SM: Sokal-Michener
    }
    return Indices


def calculate_medoid(fpe, n_ary="RR"):
    index = np.argmin(calculate_comp_sim(fpe, n_ary=n_ary)["index"])
    return fpe.fps[index][0]


def calculate_outlier(fpe, n_ary="RR"):
    index = np.argmax(calculate_comp_sim(fpe, n_ary=n_ary)["index"])
    return fpe.fps[index][0]


def calculate_comp_sim(fpe, n_ary="RR"):
    """Calculate the complementary similarity for RR, JT, or SM

    Arguments
    ---------
     fpe: FPSim2 engine
         FPSim2 engine

    n_ary : str
        String with the initials of the desired similarity index to calculate the iSIM from.
        Only RR, JT, or SM are available. For other indexes use gen_sim_dict.

    Returns
    -------
    comp_sims : nd.array
        1D array with the complementary similarities of all the molecules in the set.
    """

    data = unpack_fpe(fpe)
    c_total = np.sum(data, axis=0)
    n_objects = fpe.fps.shape[0] - 1
    m = len(c_total)

    comp_matrix = c_total - data
    a = comp_matrix * (comp_matrix - 1) / 2

    if n_ary == "RR":
        comp_sims = np.sum(a, axis=1) / (m * n_objects * (n_objects - 1) / 2)

    elif n_ary == "JT":
        comp_sims = np.sum(a, axis=1) / np.sum(
            (a + comp_matrix * (n_objects - comp_matrix)), axis=1
        )

    elif n_ary == "SM":
        comp_sims = np.sum(
            (a + (n_objects - comp_matrix) * (n_objects - comp_matrix - 1) / 2), axis=1
        ) / (m * n_objects * (n_objects - 1) / 2)
    return np.rec.fromarrays([fpe.fps[:, 0], comp_sims], names=("mol_id", "index"))
