import numpy as np
from PIL import Image

from pyencoder import HuffmanCoding, RunLengthEncoding

from utils import chunkify, get_padded_size
from dct import idct_1D, dct_1D
from lookup_table import LUMA_QUANTIZATION_TABLE, CHROMA_QUANTIZATION_TABLE


SEP = "\0"

# TODO:
# 1. encode the size of image
# 2. decide whetehr to trim the image before saving or after decompression
# 3: fix decompress function

# TODO: add zigzag runlength for the encoding process

# ! runlength string has depreciated
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
                for index, row in enumerate(block):
                    block[index] = dct_1D(row)

                for index, col in enumerate(block.T):
                    block.T[index] = dct_1D(col)

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
