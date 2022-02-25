import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import as_strided as ast

from pyencoder import HuffmanCoding, RunLengthEncoding

from lookup_table import NORMALIZED_COSINE_TABLE, WEIGHTAGE_TABLE, LUMA_QUANTIZATION_TABLE, CHROMA_QUANTIZATION_TABLE


SEP = "\0"

# TODO:
# 1. encode the size of image
# 2. decide whetehr to trim the image before saving or after decompression
# 3. optimise dct using numpy built-in functions
# 4: fix decompress function


def compress(path_to_img: str, filename: str = "") -> None:
    # Colorspace conversion
    img_arr = np.asarray(Image.open(path_to_img).convert("YCbCr"), dtype=np.float32)

    # Padding
    img_row, img_col, _ = img_arr.shape
    row_to_pad, col_to_pad = 8 - img_row % 8, 8 - img_col % 8
    img_arr: np.ndarray = np.pad(img_arr, pad_width=((0, row_to_pad), (0, col_to_pad), (0, 0)), mode="edge")

    # seperate luma & chroma for preprocessing
    Luma, CbCr = img_arr[:, :, 0], img_arr[:, :, 1:]

    # Chroma Downsampling
    row, col, _ = CbCr.shape
    shrunk_row, shrunk_col = row // 2, col // 2
    row_bin, col_bin = row // shrunk_row, col // shrunk_col
    CbCr: np.ndarray = CbCr.reshape((shrunk_row, row_bin, shrunk_col, col_bin, 2)).max(3).max(1)

    # Chroma re-Padding
    row_to_pad, col_to_pad = 8 - CbCr.shape[0] % 8, 8 - CbCr.shape[1] % 8
    CbCr = np.pad(CbCr, pad_width=((0, row_to_pad), (0, col_to_pad), (0, 0)), mode="edge")

    # seperate chroma into chroma-red & chroma-blue for dct & quantization
    ChromaB, ChromaR = CbCr[:, :, 0], CbCr[:, :, 1]

    runlengthstring = ""
    quantization_table = LUMA_QUANTIZATION_TABLE
    for n, color_channel in enumerate([Luma, ChromaB, ChromaR]):
        # divide array into equal 8x8 chunk piece
        block_arr = chunkify(color_channel, (8, 8))

        # shift the colorspace value
        color_channel -= 128

        # apply dct for each 8x8 chunk
        for blocks in block_arr:
            for block in blocks:
                dct2D_ip(block)

        # quantize the entire array at once
        scaled_quantization_table = np.tile(quantization_table, (block_arr.shape[0], block_arr.shape[1], 1, 1))
        block_arr /= scaled_quantization_table
        block_arr.round(decimals=0, out=block_arr)

        # encode the value of the array
        runlengthstring += RunLengthEncoding.encode(color_channel.astype(np.int8).flatten(), int)
        quantization_table = CHROMA_QUANTIZATION_TABLE
        runlengthstring += SEP

    img_data = f"{img_row},{img_col}" + SEP + runlengthstring[:-1]
    filename = filename if filename != "" else path_to_img.split("/")[-1].split(".")[0]
    with open(f"{filename}.test", "wb") as f:
        HuffmanCoding.dump(img_data, str, f)


def decompress(filepath: str) -> Image.Image:
    with open(filepath, "rb") as f:
        # decode the binary file
        img_size, _, img_data = HuffmanCoding.load(f).partition(SEP)

    # determine the encoded size of the 3 color channels
    img_size = list(int(s) for s in img_size.split(","))
    luma_size = get_padded_size(img_size, 8)
    chroma_size = get_padded_size((luma_size[0] // 2, luma_size[1] // 2), 8)

    # decode the color channels and reshape it to its original shape
    Luma, ChromaB, ChromaR = (np.array(RunLengthEncoding.decode(raw_data, int)) for raw_data in img_data.split(SEP))
    Luma, ChromaB, ChromaR = np.reshape(Luma, luma_size), np.reshape(ChromaB, chroma_size), np.reshape(ChromaR, chroma_size)

    quantization_table = LUMA_QUANTIZATION_TABLE
    for color_channel in [Luma, ChromaB, ChromaR]:

        # divide array into equal 8x8 chunk piece
        block_arr = chunkify(color_channel, (8, 8))

        # quantize the entire array at once
        scaled_quantization_table = np.tile(quantization_table, (block_arr.shape[0], block_arr.shape[1], 1, 1))
        block_arr *= scaled_quantization_table

        # apply inverse dct for each 8x8 chunk
        for blocks in block_arr:
            for block in blocks:
                idct2D_ip(block)

        # shift the colorspace value
        color_channel += 128

        quantization_table = CHROMA_QUANTIZATION_TABLE

    # Trim the padded chroma
    chroma_row, chroma_col = (d // 2 for d in luma_size)
    ChromaB, ChromaR = ChromaB[:chroma_row, :chroma_col], ChromaR[:chroma_row, :chroma_col]

    # duplicate the element to 'stretch' it
    ChromaB = ChromaB.repeat(2, axis=1).repeat(2, axis=0)
    ChromaR = ChromaR.repeat(2, axis=1).repeat(2, axis=0)

    # merge the 3 color channel and convert it back into RGB
    YCC = np.dstack((Luma, ChromaB, ChromaR))[: img_size[0], : img_size[1]].astype(np.uint8)

    return Image.fromarray(YCC, "YCbCr").convert("RGB")


def chunkify(arr: np.ndarray, chunk_size: tuple):
    shape = (arr.shape[0] // chunk_size[0], arr.shape[1] // chunk_size[1]) + chunk_size
    strides = (chunk_size[0] * arr.strides[0], chunk_size[1] * arr.strides[1]) + arr.strides
    return ast(arr, shape=shape, strides=strides)


def get_padded_size(arr_shape: tuple, pad_to: int) -> tuple:
    return (arr_shape[0] + (pad_to - arr_shape[0] % pad_to), arr_shape[1] + (pad_to - arr_shape[1] % pad_to))


def dct2D_ip(block: np.ndarray) -> None:
    for index, row in enumerate(block):
        block[index] = dct_1D(row)

    for index, col in enumerate(block.T):
        block.T[index] = dct_1D(col)


def idct2D_ip(block: np.ndarray) -> None:
    for index, row in enumerate(block):
        block[index] = idct_1D(row)

    for index, col in enumerate(block.T):
        block.T[index] = idct_1D(col)

    np.round(block, decimals=0, out=block)


def dct_1D(vector):
    v0 = vector[0] + vector[7]
    v1 = vector[1] + vector[6]
    v2 = vector[2] + vector[5]
    v3 = vector[3] + vector[4]
    v4 = vector[3] - vector[4]
    v5 = vector[2] - vector[5]
    v6 = vector[1] - vector[6]
    v7 = vector[0] - vector[7]

    v8 = v0 + v3
    v9 = v1 + v2
    v10 = v1 - v2
    v11 = v0 - v3
    v12 = -v4 - v5
    v13 = (v5 + v6) * WEIGHTAGE_TABLE[2]
    v14 = v6 + v7

    v15 = v8 + v9
    v16 = v8 - v9
    v17 = (v10 + v11) * WEIGHTAGE_TABLE[0]
    v18 = (v12 + v14) * WEIGHTAGE_TABLE[4]

    v19 = -v12 * WEIGHTAGE_TABLE[1] - v18
    v20 = v14 * WEIGHTAGE_TABLE[3] - v18

    v21 = v17 + v11
    v22 = v11 - v17
    v23 = v13 + v7
    v24 = v7 - v13

    v25 = v19 + v24
    v26 = v23 + v20
    v27 = v23 - v20
    v28 = v24 - v19

    return [
        NORMALIZED_COSINE_TABLE[0] * v15,
        NORMALIZED_COSINE_TABLE[1] * v26,
        NORMALIZED_COSINE_TABLE[2] * v21,
        NORMALIZED_COSINE_TABLE[3] * v28,
        NORMALIZED_COSINE_TABLE[4] * v16,
        NORMALIZED_COSINE_TABLE[5] * v25,
        NORMALIZED_COSINE_TABLE[6] * v22,
        NORMALIZED_COSINE_TABLE[7] * v27,
    ]


def idct_1D(vector):
    v15 = vector[0] / NORMALIZED_COSINE_TABLE[0]
    v26 = vector[1] / NORMALIZED_COSINE_TABLE[1]
    v21 = vector[2] / NORMALIZED_COSINE_TABLE[2]
    v28 = vector[3] / NORMALIZED_COSINE_TABLE[3]
    v16 = vector[4] / NORMALIZED_COSINE_TABLE[4]
    v25 = vector[5] / NORMALIZED_COSINE_TABLE[5]
    v22 = vector[6] / NORMALIZED_COSINE_TABLE[6]
    v27 = vector[7] / NORMALIZED_COSINE_TABLE[7]

    v19 = (v25 - v28) / 2
    v20 = (v26 - v27) / 2
    v23 = (v26 + v27) / 2
    v24 = (v25 + v28) / 2

    v7 = (v23 + v24) / 2
    v11 = (v21 + v22) / 2
    v13 = (v23 - v24) / 2
    v17 = (v21 - v22) / 2

    v8 = (v15 + v16) / 2
    v9 = (v15 - v16) / 2

    v18 = (v19 - v20) * WEIGHTAGE_TABLE[4]  # Different from original
    v12 = (v19 * WEIGHTAGE_TABLE[3] - v18) / (
        WEIGHTAGE_TABLE[1] * WEIGHTAGE_TABLE[4] - WEIGHTAGE_TABLE[1] * WEIGHTAGE_TABLE[3] - WEIGHTAGE_TABLE[3] * WEIGHTAGE_TABLE[4]
    )
    v14 = (v18 - v20 * WEIGHTAGE_TABLE[1]) / (
        WEIGHTAGE_TABLE[1] * WEIGHTAGE_TABLE[4] - WEIGHTAGE_TABLE[1] * WEIGHTAGE_TABLE[3] - WEIGHTAGE_TABLE[3] * WEIGHTAGE_TABLE[4]
    )

    v6 = v14 - v7
    v5 = v13 / WEIGHTAGE_TABLE[2] - v6
    v4 = -v5 - v12
    v10 = v17 / WEIGHTAGE_TABLE[0] - v11

    v0 = (v8 + v11) / 2
    v1 = (v9 + v10) / 2
    v2 = (v9 - v10) / 2
    v3 = (v8 - v11) / 2

    return [
        (v0 + v7) / 2,
        (v1 + v6) / 2,
        (v2 + v5) / 2,
        (v3 + v4) / 2,
        (v3 - v4) / 2,
        (v2 - v5) / 2,
        (v1 - v6) / 2,
        (v0 - v7) / 2,
    ]
