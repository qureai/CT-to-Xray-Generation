import numpy as np
from readNPYheader import readNPYheader 
def readNPY(filename):
    # Function to read NPY files into Python.
    # Only reads a subset of all possible NPY files, specifically N-D arrays of certain data types.

    # Read header information
    shape, dataType, fortranOrder, littleEndian, totalHeaderLength, _ = readNPYheader(filename)

    # Open the file for reading
    if littleEndian:
        fid = open(filename, 'rb')
    else:
        fid = open(filename, 'rb')

    try:
        # Skip the header
        fid.read(totalHeaderLength)

        # Read the data
        data = np.fromfile(fid, dtype=np.dtype(dataType))
        data = data.reshape(shape)

        if len(shape) > 1 and not fortranOrder:
            data = np.reshape(data, shape[::-1])
            data = np.transpose(data, axes=list(range(len(shape)-1, -1, -1)))
        elif len(shape) > 1:
            data = np.reshape(data, shape)

        return data

    finally:
        fid.close()

# Example usage:
# data = readNPY('your_file.npy')
