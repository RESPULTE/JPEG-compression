from turtle import color
import numpy as np
from PIL import Image

from pyencoder import HuffmanCoding, RunLengthEncoding
from pyencoder.utils import zigzag

from pyimagery.utils import chunkify, get_padded_size
from pyimagery.jpeg.dct import idct_1D, dct_1D
from pyimagery.jpeg.lookup_table import LUMA_QUANTIZATION_TABLE, CHROMA_QUANTIZATION_TABLE

SEP = "\0"

# TODO:
# 1. encode the size of image
# 2. decide whetehr to trim the image before saving or after decompression
# 3: fix decompress function

# TODO: add zigzag runlength for the encoding process


def compress(path_to_img: str) -> None:
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

    # for storage purpose
    color_channel_name = ["Luma", "ChromaB", "ChromaR"]
    headers = {c: {"dc": "", "ac": ""} for c in color_channel_name}
    encoded_data = {c: {"dc": "", "ac": ""} for c in color_channel_name}

    to_process = [
        (Luma, LUMA_QUANTIZATION_TABLE),
        (ChromaB, CHROMA_QUANTIZATION_TABLE),
        (ChromaR, CHROMA_QUANTIZATION_TABLE),
    ]
    for n, (color_channel, quantization_table) in enumerate(to_process):
        # divide array into equal 8x8 chunk piece
        chunk_view = chunkify(color_channel, (8, 8))
        chunk_view_row, chunk_view_col = chunk_view.shape[0], chunk_view.shape[1]

        # shift the colorspace value
        color_channel -= 128

        # apply dct for each 8x8 chunk
        for chunks in chunk_view:
            for chunk in chunks:
                for index, row in enumerate(chunk):
                    chunk[index] = dct_1D(row)

                for index, col in enumerate(chunk.T):
                    chunk.T[index] = dct_1D(col)

        # quantize the entire array at once
        scaled_quantization_table = np.tile(quantization_table, (chunk_view_row, chunk_view_col, 1, 1))
        chunk_view /= scaled_quantization_table
        chunk_view = chunk_view.astype(np.int64, copy=False)

        # encode the DC/AC value
        total_chunk_in_arr = chunk_view_row * chunk_view_col
        dc_values = np.zeros(total_chunk_in_arr, dtype=np.int64)
        ac_values = []

        for i, blocks in enumerate(chunk_view):
            for j, block in enumerate(blocks):
                index = i * chunk_view_col + j
                ac_values.extend(RunLengthEncoding.encode(zigzag(block, "d")[1:], target_values=[0]))

                # to be removed
                if index == 0:
                    dc_values[0] = block[0, 0]
                    continue

                dc_values[index] = block[0, 0] - blocks[j - 1, 0, 0]

        dc_codebook, encoded_dc_values = HuffmanCoding.encode(dc_values)
        ac_codebook, encoded_ac_values = HuffmanCoding.encode(ac_values)

        dc_header = HuffmanCoding.generate_header_from_codebook(dc_codebook, "i")
        ac_header = HuffmanCoding.generate_header_from_codebook(ac_codebook, "i")

        curr_channel_name = color_channel_name[n]

        headers[curr_channel_name]["dc"] = dc_header
        headers[curr_channel_name]["ac"] = ac_header

        encoded_data[curr_channel_name]["dc"] = encoded_dc_values
        encoded_data[curr_channel_name]["ac"] = encoded_ac_values


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
    Luma, ChromaB, ChromaR = (
        np.reshape(Luma, luma_size),
        np.reshape(ChromaB, chroma_size),
        np.reshape(ChromaR, chroma_size),
    )

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
                for index, row in enumerate(block):
                    block[index] = idct_1D(row)

                for index, col in enumerate(block.T):
                    block.T[index] = idct_1D(col)

                np.round(block, decimals=0, out=block)

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


# compress("C:/Users/yeapz/OneDrive/Desktop/Python/PyImagery/src/pyimagery/jpeg/test.png")
