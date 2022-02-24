from calendar import c
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

from PIL import Image

from pyencoder import HuffmanCoding, RunLengthEncoding

COSINE_TABLE = [
    [
        1.0,
        0.9807852804032304,
        0.9238795325112867,
        0.8314696123025452,
        0.7071067811865476,
        0.5555702330196023,
        0.38268343236508984,
        0.19509032201612833,
    ],
    [
        1.0,
        0.8314696123025452,
        0.38268343236508984,
        -0.1950903220161282,
        -0.7071067811865475,
        -0.9807852804032304,
        -0.9238795325112868,
        -0.5555702330196022,
    ],
    [
        1.0,
        0.5555702330196023,
        -0.3826834323650897,
        -0.9807852804032304,
        -0.7071067811865477,
        0.1950903220161283,
        0.9238795325112865,
        0.8314696123025456,
    ],
    [
        1.0,
        0.19509032201612833,
        -0.9238795325112867,
        -0.5555702330196022,
        0.7071067811865474,
        0.8314696123025456,
        -0.3826834323650899,
        -0.9807852804032307,
    ],
    [
        1.0,
        -0.1950903220161282,
        -0.9238795325112868,
        0.5555702330196018,
        0.7071067811865477,
        -0.8314696123025451,
        -0.38268343236509056,
        0.9807852804032305,
    ],
    [
        1.0,
        -0.555570233019602,
        -0.38268343236509034,
        0.9807852804032304,
        -0.7071067811865467,
        -0.19509032201612803,
        0.9238795325112867,
        -0.831469612302545,
    ],
    [
        1.0,
        -0.8314696123025453,
        0.38268343236509,
        0.19509032201612878,
        -0.7071067811865471,
        0.9807852804032307,
        -0.9238795325112864,
        0.5555702330196015,
    ],
    [
        1.0,
        -0.9807852804032304,
        0.9238795325112865,
        -0.8314696123025451,
        0.7071067811865466,
        -0.5555702330196015,
        0.38268343236508956,
        -0.19509032201612858,
    ],
]

MATRIX_RANGE = [(i, j) for i in range(8) for j in range(8)]
INV_ROOT2 = 1 / math.sqrt(2)

SOF = (65496).to_bytes(2, "big")
EOF = (65497).to_bytes(2, "big")
MARKER = (255).to_bytes(1, "big")

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


def compress(path_to_img: str, filename: str = "") -> None:
    # Colorspace conversion
    img_arr = np.asarray(Image.open(path_to_img).convert("YCbCr"), dtype=np.float32)

    # Padding
    row_to_pad, col_to_pad = 8 - img_arr.shape[0] % 8, 8 - img_arr.shape[1] % 8
    img_arr: np.ndarray = np.pad(img_arr, pad_width=((0, row_to_pad), (0, col_to_pad), (0, 0)), mode="edge")

    Luma, CbCr = img_arr[:, :, 0], img_arr[:, :, 1:]

    # Chroma Downsampling
    row, col, _ = CbCr.shape
    shrunk_row, shrunk_col = row // 2, col // 2
    row_bin, col_bin = row // shrunk_row, col // shrunk_col
    CbCr = CbCr.reshape((shrunk_row, row_bin, shrunk_col, col_bin, 2)).max(3).max(1)

    # Chroma re-Padding
    row_to_pad, col_to_pad = 8 - CbCr.shape[0] % 8, 8 - CbCr.shape[1] % 8
    CbCr = np.pad(CbCr, pad_width=((0, row_to_pad), (0, col_to_pad), (0, 0)), mode="edge")

    ChromaB, ChromaR = CbCr[:, :, 0], CbCr[:, :, 1]

    # TODO: set a EOF to signify the 'edge' of image
    runlengthstring = ""
    quantization_table = LUMA_QUANTIZATION_TABLE
    for color_channel in [Luma, ChromaB, ChromaR]:
        # shift the colorspace value
        color_channel -= 128
        # divide array into equal 8x8 chunk piece
        block_arr = chunkify(color_channel, (8, 8))

        # apply dct for each 8x8 chunk
        for blocks in block_arr:
            for block in blocks:
                dct2D_ip(block)

        # quantize the entire array at once
        scaled_quantization_table = np.broadcast_to(quantization_table, block_arr.shape)
        block_arr //= scaled_quantization_table

        # merge the 8x8 chunk piece back
        block_arr = unchunkify(block_arr, color_channel.shape).astype(np.int8)

        # encode the value of the array
        runlengthstring += RunLengthEncoding.encode(block_arr, int, "dz")
        quantization_table = CHROMA_QUANTIZATION_TABLE
        runlengthstring += "\0"
    p = HuffmanCoding.encode(runlengthstring, str)
    print(len(p[1]) + len(p[0]))
    # filename = filename if filename != "" else path_to_img.split("/")[-1].split(".")[0]
    # with open(f"{filename}.test", "wb") as f:
    #     HuffmanCoding.dump(runlengthstring[:-1], str, f)


def decompress(filepath: str):
    with open(filepath, "rb") as f:
        imgdata = HuffmanCoding.load(f)
        color_arr = []
        for color_channel in imgdata.split("\0"):
            decoded_color_channel = np.array(RunLengthEncoding.decode(color_channel, int)).reshape(-1, 64)
            total_curr_channel_arr = np.zeros(decoded_color_channel.shape[0])

            for i, chunk in enumerate(decoded_color_channel):
                # TODO:
                # reverse the zigzag traversal
                # form 8x8 matrix, use hstack to merge chunks until EOF is reached
                # -> new_row, continue
                # dimensionn of color 'total_curr_channel_arr' should be 2D throughout the process
                chunkified_color_arr_section = None
                total_curr_channel_arr[i] = chunkified_color_arr_section

            color_arr.append(total_curr_channel_arr)

        return np.stack(color_arr, axis=2)


def chunkify(arr: np.ndarray, chunk_size: tuple):
    shape = (arr.shape[0] // chunk_size[0], arr.shape[1] // chunk_size[1]) + chunk_size
    strides = (chunk_size[0] * arr.strides[0], chunk_size[1] * arr.strides[1]) + arr.strides
    return ast(arr, shape=shape, strides=strides)


def dct2D_ip(block: np.ndarray) -> None:
    lookup_arr = block.copy()
    for (u, v) in MATRIX_RANGE:
        block[u, v] = sum(lookup_arr[x][y] * COSINE_TABLE[x][u] * COSINE_TABLE[y][v] for (x, y) in MATRIX_RANGE)
    block[0, :] *= INV_ROOT2
    block[:, 0] *= INV_ROOT2
    block *= 0.25


def unchunkify(arr: np.ndarray, newshape: tuple):
    return arr.transpose(2, 0, 1, 3).swapaxes(0, 1).reshape(*newshape)


import cProfile

# cProfile.run('compress("test.jpg")', sort="cumtime")
compress("test.jpg")
