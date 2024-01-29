import numpy as np

def unpack_fpe(fpe):
    # need to unpack the fingerprints for iSIM functions
    data = fpe.fps[:, 1:-1].view("uint8")
    bits = np.unpackbits(data[:, np.newaxis], axis=1).ravel()
    data = bits.reshape(int(bits.size / (data.shape[1] * 8)), data.shape[1] * 8)
    return data

def get_c_total_from_fpe(fpe):
    data = np.zeros(((fpe.fps.shape[1] - 2) * 64), dtype="uint64")
    for m in fpe.fps[:, 1:-1].view("uint8"):
        data += np.unpackbits(m[:, np.newaxis], axis=1).ravel()
    return data
