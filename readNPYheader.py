import struct
import re

def readNPYheader(filename):
    # Open the file in binary mode
    with open(filename, 'rb') as f:
        # Verify that the file exists
        if not f:
            raise FileNotFoundError(f"File not found: {filename}")

        try:
            # Read the magic string
            magic_string = f.read(6)
            
            if magic_string != b'\x93NUMPY':
                raise ValueError("Error: This file does not appear to be NUMPY format based on the header.")

            # Read the version
            major_version, minor_version = struct.unpack('<BB', f.read(2))
            npy_version = (major_version, minor_version)

            # Read the header length
            header_length = struct.unpack('<H', f.read(2))[0]

            total_header_length = 10 + header_length

            # Read the array format
            array_format = f.read(header_length).decode('ascii')

            # Interpret the array format
            dtype_np = re.search(r"'descr'\s*:\s*'(.*?)'", array_format).group(1)
            little_endian = dtype_np[0] != '>'
            data_type = {
                'u1': 'uint8', 'u2': 'uint16', 'u4': 'uint32', 'u8': 'uint64',
                'i1': 'int8', 'i2': 'int16', 'i4': 'int32', 'i8': 'int64',
                'f4': 'float32', 'f8': 'float64', 'b1': 'bool'
            }[dtype_np[1:]]
            
            fortran_order = 'True' in re.search(r"'fortran_order'\s*:\s*(\w+)", array_format).group(1)

            shape_str = re.search(r"'shape'\s*:\s*\((.*?)\)", array_format).group(1)
            array_shape = [int(dim) for dim in shape_str.split(',')]

            return array_shape, data_type, fortran_order, little_endian, total_header_length, npy_version

        except Exception as e:
            raise e
