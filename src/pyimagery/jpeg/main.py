from typing import Callable, List, Tuple

from PIL import Image
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

from pyimagery.jpeg import huffman
from pyimagery.jpeg import runlength
from pyimagery.jpeg.dct import idct_1D, dct_1D
from pyimagery.jpeg.lookup_table import LUMA_QUANTIZATION_TABLE, CHROMA_QUANTIZATION_TABLE

SEP = "\0"

# TODO:
# 1. encode the size of image
# 2. decide whetehr to trim the image before saving or after decompression
# 3: fix decompress function
# 4: make quantization table's weightage 'settable'


def pad_to(arr: np.ndarray, row_mult: int, col_mult: int, mode: str = "edge") -> np.ndarray:
    row_to_pad, col_to_pad = row_mult - arr.shape[0] % row_mult, col_mult - arr.shape[1] % col_mult
    return np.pad(arr, pad_width=((0, row_to_pad), (0, col_to_pad), (0, 0)), mode=mode)


def downsample(arr: np.ndarray, ratio: Tuple[int, int] = (2, 2)) -> np.ndarray:
    row, col = arr.shape[0], arr.shape[1]
    shrunk_row, shrunk_col = row // ratio[0], col // ratio[1]
    row_bin, col_bin = row // shrunk_row, col // shrunk_col
    return np.reshape(arr, (shrunk_row, row_bin, shrunk_col, col_bin, 2)).max(3).max(1)


def chunkify(arr: np.ndarray, chunk_size: tuple):
    shape = (arr.shape[0] // chunk_size[0], arr.shape[1] // chunk_size[1]) + chunk_size
    strides = (chunk_size[0] * arr.strides[0], chunk_size[1] * arr.strides[1]) + arr.strides
    return ast(arr, shape=shape, strides=strides)


def _preprocess(img_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # pad the image array to be a multiple of 8
    img_arr = pad_to(img_arr, 8, 8, mode="edge")

    # seperate luma & chroma for preprocessing
    Luma, CbCr = img_arr[:, :, 0], img_arr[:, :, 1:]

    # pad the chroma again to fit an 8x8 matrix
    CbCr = pad_to(downsample(CbCr, ratio=(2, 2)), 8, 8, mode="edge")

    # seperate chroma into chroma-red & chroma-blue for dct & quantization
    ChromaB, ChromaR = CbCr[:, :, 0], CbCr[:, :, 1]

    return Luma, ChromaB, ChromaR


def compress(path_to_img: str) -> None:
    # Colorspace conversion
    img_arr = np.asarray(Image.open(path_to_img).convert("YCbCr"), dtype=np.int64)

    Luma, ChromaB, ChromaR = _preprocess(img_arr)

    YCbCr_rawdata = {"Luma": {"AC": [], "DC": []}, "Chroma": {"AC": [], "DC": []}}
    to_process = [
        (Luma, "Luma", LUMA_QUANTIZATION_TABLE),
        (ChromaB, "Chroma", CHROMA_QUANTIZATION_TABLE),
        (ChromaR, "Chroma", CHROMA_QUANTIZATION_TABLE),
    ]
    for color_channel, channel_type, quantization_table in to_process:
        # shift the colorspace value
        color_channel -= 128

        # get a view of the array in 8x8 equal chunk piece
        chunk_view = chunkify(color_channel, (8, 8))
        chunk_view_row, chunk_view_col = chunk_view.shape[0], chunk_view.shape[1]

        # for temporaray storage of the data
        total_chunks = chunk_view_row * chunk_view_col
        dc_values = np.zeros(total_chunks, dtype=np.int64)
        ac_values = []

        # loop through the aray view in 8x8 chunks
        for i, chunks in enumerate(chunk_view):

            # as the initial value for the differential encoding
            dc_values[0] = chunks[0, 0, 0]
            for j, chunk in enumerate(chunks):

                # 2D discrete cosine transform
                for index, row in enumerate(chunk):
                    chunk[index] = dct_1D(row)

                for index, col in enumerate(chunk.T):
                    chunk.T[index] = dct_1D(col)

                # quantizatize the chunk
                np.floor_divide(chunk, quantization_table, out=chunk)

                # encode it
                chunk_index = i * chunk_view_col + j
                dc_values[chunk_index] = chunk[0, 0] - chunks[j - 1, 0, 0]
                ac_values.extend(runlength.encode(runlength.zigzag(chunk)[1:]))

        # using extend as the DC/AC values for the ChromaR and ChromaB are grouped together in the encoding process
        YCbCr_rawdata[channel_type]["DC"].extend(dc_values)
        YCbCr_rawdata[channel_type]["AC"].extend(ac_values)

    YCbCr_encoded_data = {"Luma": {"AC": [], "DC": []}, "Chroma": {"AC": [], "DC": []}}

    # the huffman encoding process for AC/DC
    for channel_type in ("Luma", "Chroma"):
        for data_type in ("DC", "AC"):
            codebook, encoded_data = huffman.encode(YCbCr_rawdata[channel_type][data_type], data_type)
            header = huffman.generate_header_from_codebook(codebook, data_type)
            YCbCr_encoded_data[channel_type][data_type] = (header, encoded_data)


# def decompress(filepath: str) -> Image.Image:
#     with open(filepath, "rb") as f:
#         # decode the binary file
#         img_size, _, img_data = HuffmanCoding.load(f).partition(SEP)

#     # determine the encoded size of the 3 color channels
#     img_size = list(int(s) for s in img_size.split(","))
#     luma_size = get_padded_size(img_size, 8)
#     chroma_size = get_padded_size((luma_size[0] // 2, luma_size[1] // 2), 8)

#     # decode the color channels and reshape it to its original shape
#     Luma, ChromaB, ChromaR = (np.array(RunLengthEncoding.decode(raw_data, int)) for raw_data in img_data.split(SEP))
#     Luma, ChromaB, ChromaR = (
#         np.reshape(Luma, luma_size),
#         np.reshape(ChromaB, chroma_size),
#         np.reshape(ChromaR, chroma_size),
#     )

#     quantization_table = LUMA_QUANTIZATION_TABLE
#     for color_channel in [Luma, ChromaB, ChromaR]:

#         # divide array into equal 8x8 chunk piece
#         block_arr = chunkify(color_channel, (8, 8))

#         # quantize the entire array at once
#         scaled_quantization_table = np.tile(quantization_table, (block_arr.shape[0], block_arr.shape[1], 1, 1))
#         block_arr *= scaled_quantization_table

#         # apply inverse dct for each 8x8 chunk
#         for blocks in block_arr:
#             for block in blocks:
#                 for index, row in enumerate(block):
#                     block[index] = idct_1D(row)

#                 for index, col in enumerate(block.T):
#                     block.T[index] = idct_1D(col)

#                 np.round(block, decimals=0, out=block)

#         # shift the colorspace value
#         color_channel += 128

#         quantization_table = CHROMA_QUANTIZATION_TABLE

#     # Trim the padded chroma
#     chroma_row, chroma_col = (d // 2 for d in luma_size)
#     ChromaB, ChromaR = ChromaB[:chroma_row, :chroma_col], ChromaR[:chroma_row, :chroma_col]

#     # duplicate the element to 'stretch' it
#     ChromaB = ChromaB.repeat(2, axis=1).repeat(2, axis=0)
#     ChromaR = ChromaR.repeat(2, axis=1).repeat(2, axis=0)

#     # merge the 3 color channel and convert it back into RGB
#     YCC = np.dstack((Luma, ChromaB, ChromaR))[: img_size[0], : img_size[1]].astype(np.uint8)

#     return Image.fromarray(YCC, "YCbCr").convert("RGB")


compress("C:/Users/yeapz/OneDrive/Desktop/Python/PyImagery/src/pyimagery/jpeg/test.png")
