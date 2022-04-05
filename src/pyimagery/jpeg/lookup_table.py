import numpy as np
from enum import Enum

LUMA_QUANTIZATION_TABLE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.int8,
)

CHROMA_QUANTIZATION_TABLE = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.int8,
)


class JPEG_MARKER(Enum):
    START_OF_FILE = 0xFFD8

    HEADER = 0xFFE0

    QUANTIZATION_TABLE = 0xFFDB

    START_OF_FRAME = 0xFFC0

    HUFFMAN_TABLE = 0xFFC4
    DC_HUFFMAN_TABLE_TYPE = 0x0
    AC_HUFFMAN_TABLE_TYPE = 0x1

    LUMA_TYPE = 0x0
    CHROMA_TYPE = 0x1

    START_OF_SCAN = 0xFFDA

    END_OF_FILE = 0xFFD9
