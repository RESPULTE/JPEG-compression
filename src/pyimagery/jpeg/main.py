from typing import Callable, Dict, List, Tuple

from PIL import Image
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

from pyimagery.jpeg import huffman, runlength
from pyimagery.jpeg.dct import idct_1D, dct_1D
from pyimagery.jpeg.lookup_table import JPEG_MARKER, LUMA_QUANTIZATION_TABLE, CHROMA_QUANTIZATION_TABLE

from pyimagery._type_hint import Bitcode
from pyimagery.utils.binary import tobytes

# TODO:
# 1. encode the size of image
# 2. decide whetehr to trim the image before saving or after decompression
# 3: fix decompress function
# 4: make quantization table's weightage 'settable'


def pad_to(arr: np.ndarray, row_mult: int, col_mult: int, mode: str = "edge") -> np.ndarray:
    row_to_pad, col_to_pad = (row_mult - arr.shape[0] % row_mult), (col_mult - arr.shape[1] % col_mult)
    return np.pad(arr, pad_width=((0, row_to_pad), (0, col_to_pad), (0, 0)), mode=mode)


def downsample(arr: np.ndarray, ratio: Tuple[int, int]) -> np.ndarray:
    row, col = arr.shape[0], arr.shape[1]
    shrunk_row, shrunk_col = row // ratio[0], col // ratio[1]
    row_bin, col_bin = row // shrunk_row, col // shrunk_col
    return np.reshape(arr, (shrunk_row, row_bin, shrunk_col, col_bin, 2)).max(3).max(1)


def chunkify(arr: np.ndarray, chunk_size: Tuple[int, ...]) -> np.ndarray:
    """
    creates a view into the array that's been cut into the specified chunk size

    Args:
        arr (np.ndarray): an array of at least 2 dimension
        chunk_size (tuple): the chunk size used for the 'cutting process'

    Returns:
        np.ndarray: a view into the given numpy array 'cut' with the specified chunk size
    """
    shape = (arr.shape[0] // chunk_size[0], arr.shape[1] // chunk_size[1]) + chunk_size
    strides = (chunk_size[0] * arr.strides[0], chunk_size[1] * arr.strides[1]) + arr.strides
    return ast(arr, shape=shape, strides=strides)


def preprocess(img_arr: np.ndarray, ratio: Tuple[int, int] = (2, 2)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    process the image array in preperation for the compression process

    Args:
        img_arr (np.ndarray): a 3D array made of color channels
        ratio (Tuple[int, int], optional): downsampling ratio. Defaults to (2, 2).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
         3 color channels seperated and processed from the given image array
    """
    # pad the image array to be a multiple of 8
    img_arr = pad_to(img_arr, 8, 8, mode="edge")

    # seperate luma & chroma for preprocessing
    Luma, CbCr = img_arr[:, :, 0], img_arr[:, :, 1:]

    # pad the chroma again to fit an 8x8 matrix
    downsampled_CbCr = pad_to(downsample(CbCr, ratio), 8, 8, mode="edge")

    # seperate chroma into chroma-red & chroma-blue for dct & quantization
    downsampled_ChromaB, downsampled_ChromaR = downsampled_CbCr[:, :, 0], downsampled_CbCr[:, :, 1]

    return Luma, downsampled_ChromaB, downsampled_ChromaR


def encode_compressed_data(YCbCr_rawdata: Dict[str, Dict[str, Bitcode]]) -> Tuple[Dict[str, Dict[str, bytes]], bytes]:
    YCbCr_huffman_table = {"Luma": {"AC": [], "DC": []}, "Chroma": {"AC": [], "DC": []}}
    YCbCr_encoded_data = ""

    for channel_type in ("Luma", "Chroma"):

        for table_type in ("DC", "AC"):

            codebook, encoded_data = huffman.encode(YCbCr_rawdata[channel_type][table_type], table_type)

            YCbCr_huffman_table[channel_type][table_type] = huffman.generate_header_from_codebook(codebook, table_type)
            YCbCr_encoded_data += encoded_data

    return YCbCr_huffman_table, tobytes(YCbCr_encoded_data, "bin", signed=False)


def assemble_quantization_table(luma_quantization_table: np.ndarray, chroma_quantization_table: np.ndarray) -> bytes:
    qtable_marker = tobytes(JPEG_MARKER.QUANTIZATION_TABLE.value, "H", 2)

    luma_qtable_data = tobytes(JPEG_MARKER.LUMA_TYPE.value, "B", 1) + luma_quantization_table.tobytes()
    chroma_qtable_data = tobytes(JPEG_MARKER.CHROMA_TYPE.value, "B", 1) + chroma_quantization_table.tobytes()

    luma_qtable_len = tobytes(len(luma_qtable_data), "H", 2)
    chroma_qtable_len = tobytes(len(chroma_qtable_data), "H", 2)

    luma_qtable_bytedata = qtable_marker + luma_qtable_len + luma_qtable_data
    chroma_qtable_bytedata = qtable_marker + chroma_qtable_len + chroma_qtable_data

    return luma_qtable_bytedata, chroma_qtable_bytedata


def assemble_huffman_table(YCbCr_huffman_table: Dict[str, Dict[str, bytes]]) -> bytes:
    htable_bytedata = bytes()

    marker = tobytes(JPEG_MARKER.HUFFMAN_TABLE.value, "H", 2)

    dc_table_class: int = JPEG_MARKER.DC_HUFFMAN_TABLE_TYPE.value
    ac_table_class: int = JPEG_MARKER.AC_HUFFMAN_TABLE_TYPE.value

    luma_table_dest: int = JPEG_MARKER.LUMA_TYPE.value
    chroma_table_dest: int = JPEG_MARKER.CHROMA_TYPE.value

    table_dest = luma_table_dest
    for channel_type in ("Luma", "Chroma"):

        table_class = dc_table_class
        for table_type in ("DC", "AC"):

            raw_table_data = YCbCr_huffman_table[channel_type][table_type]

            # some bitwise trickery to combine the class & destination of table in 1 byte
            table_type = (table_class << table_dest.bit_length()) + table_dest
            table_data = tobytes(table_type, "B", 1) + tobytes(raw_table_data, "bin")

            # assembling the header
            table_length = tobytes(len(table_data) + 2, "H", 2)
            header = marker + table_length + table_data

            htable_bytedata += header

            table_class = ac_table_class

        table_dest = chroma_table_dest

    return htable_bytedata


# TODO: extract to a proper config file
# TODO: what the fuck are all of these for
def assemble_start_of_frame(
    luma_downsampling_factor: Tuple[int, int],
    Cb_downsampling_factor: Tuple[int, int],
    Cr_downsampling_factor: Tuple[int, int],
) -> bytes:
    sof_marker = tobytes(JPEG_MARKER.START_OF_FRAME, "H", 2)

    percision = 8
    line_Nb = 2
    samples = 6
    components = 3

    chromaB_dfactor = (Cb_downsampling_factor[0] << Cb_downsampling_factor[1].bit_length()) + Cb_downsampling_factor[1]

    chromaR_dfactor = (Cr_downsampling_factor[0] << Cr_downsampling_factor[1].bit_length()) + Cr_downsampling_factor[1]

    luma_dfactor = (
        (luma_downsampling_factor[0] << luma_downsampling_factor[1].bit_length()) + luma_downsampling_factor[1],
    )

    sof_data = tobytes(
        percision,
        line_Nb,
        samples,
        components,
        1,
        luma_dfactor,
        0,
        2,
        chromaB_dfactor,
        1,
        3,
        chromaR_dfactor,
        1,
    )

    sof_len = tobytes(len(sof_data) + 2, "H", 2)
    return sof_marker + sof_len + sof_data


# TODO: extract to a proper config file
# TODO: what the fuck are all of these for
def assemble_start_of_scan() -> bytes:
    sos_marker = tobytes(JPEG_MARKER.START_OF_SCAN.value, "H", 2)
    component_num = 3

    Y_specification = (0 << int(0).bit_length()) + 0
    Cb_specification = (1 << int(1).bit_length()) + 1
    Cr_specification = (1 << int(1).bit_length()) + 1

    stuff_1 = tobytes([component_num, 1, Y_specification, 2, Cb_specification, 3, Cr_specification], "B", 1)
    stuff_2 = tobytes([0, 63, 1], "B", 1)

    stuff = stuff_1 + stuff_2
    sos_len = tobytes(len(stuff) + 2, "H", 2)
    return sos_marker + sos_len + stuff


# TODO: extract to a proper config file
# TODO: what the fuck are all of these for
def assemble_header() -> bytes:
    header_marker = tobytes(JPEG_MARKER.HEADER.value, "H", 2)

    # identifier, version(1.1), units(1)
    stuff = ".J .F .I .F 00" + tobytes([1, 1, 1], "B", 1)
    density = tobytes([72, 72], "H", 2)

    thumbnail = tobytes([0, 0], "B", 1)

    header = stuff + density + thumbnail
    header_len = tobytes(len(header) + 2, "H", 2)

    return header_marker + header_len + header


# TODO: possibly getting this its own dedicated class to work with
def assemble(
    luma_quantization_table: np.ndarray,
    chroma_quantization_table: np.ndarray,
    Y_downsampling_factor: Tuple[int, int],
    Cb_downsampling_factor: Tuple[int, int],
    Cr_downsampling_factor: Tuple[int, int],
    YCbCr_huffman_table: Dict[str, Dict[str, Bitcode]],
    YCbCr_encoded_data: Bitcode,
) -> bytes:
    """
    convert the 'active data' of the jpeg compression process into bytes
    and then add the various markers to them

    Args:
        luma_quantization_table (np.ndarray)
        chroma_quantization_table (np.ndarray)
        YCbCr_huffman_table (Dict[str, Dict[str, Bitcode]])
        YCbCr_encoded_data (Bitcode)

    Returns:
        bytes
    """
    jpeg_datapack = bytes()

    jpeg_datapack += tobytes(JPEG_MARKER.START_OF_FILE.value, "H", 2)
    jpeg_datapack += assemble_header()
    jpeg_datapack += assemble_quantization_table(luma_quantization_table, chroma_quantization_table)
    jpeg_datapack += assemble_start_of_frame(Y_downsampling_factor, Cb_downsampling_factor, Cr_downsampling_factor)
    jpeg_datapack += assemble_huffman_table(YCbCr_huffman_table)
    jpeg_datapack += assemble_start_of_scan()
    jpeg_datapack += tobytes(YCbCr_encoded_data, "bin")
    jpeg_datapack += tobytes(JPEG_MARKER.END_OF_FILE.value, "H", 2)

    return jpeg_datapack


def compress(path_to_img: str, *, compression_ratio: int = 1, downsample_ratio: Tuple[int, int] = (2, 2)) -> None:
    # Colorspace conversion
    img_arr = np.asarray(Image.open(path_to_img).convert("YCbCr"), dtype=np.int64)

    # downsampling, padding and other misc. stuff happens here
    Luma, ChromaB, ChromaR = preprocess(img_arr, downsample_ratio)

    # increase/decrease the compression ratio based on the given integer
    luma_qtable = np.floor_divide(LUMA_QUANTIZATION_TABLE, compression_ratio)
    chroma_qtable = np.floor_divide(CHROMA_QUANTIZATION_TABLE, compression_ratio)

    # used for the for-loop below as temporary data container
    YCbCr_rawdata = {"Luma": {"AC": [], "DC": []}, "Chroma": {"AC": [], "DC": []}}
    quantization_table = luma_qtable
    channel_type = "Luma"

    for color_channel in (Luma, ChromaB, ChromaR):

        # shift the colorspace value
        color_channel -= 128

        # get a view of the array in 8x8 equal chunk piece
        chunk_view = chunkify(color_channel, (8, 8))
        chunk_view_row, chunk_view_col = chunk_view.shape[0], chunk_view.shape[1]

        # using a normal list instead of numpy's array as
        # the return value from RLE varies wildly
        total_chunks = chunk_view_row * chunk_view_col
        dc_values = np.zeros(total_chunks, dtype=np.int64)
        ac_values = []

        # loop through the aray view in 8x8 chunks
        chunk_index = 0
        for chunks in chunk_view:

            for j, chunk in enumerate(chunks, 1):

                # 2D discrete cosine transform
                for index, row in enumerate(chunk):
                    chunk[index] = dct_1D(row)

                for index, col in enumerate(chunk.T):
                    chunk.T[index] = dct_1D(col)

                # quantizatize the chunk
                np.floor_divide(chunk, quantization_table, out=chunk)

                # encode DC with Delta encoding
                dc_values[chunk_index] = chunk[0, 0] - chunks[j - 1, 0, 0]

                # encode AC with Runlength encoding
                ac_values.extend(runlength.encode(runlength.zigzag(chunk)[1:]))

                chunk_index += 1

        # reset the initial value for the differential encoding
        # since the 'j' index for the initial value would be -1,
        #  grabbing the last chunk when it shouldn't have gotten any
        dc_values[0] = chunks[0, 0, 0]

        # using extend as the DC/AC values for the ChromaR and ChromaB are grouped together in the encoding process
        YCbCr_rawdata[channel_type]["DC"].extend(dc_values)
        YCbCr_rawdata[channel_type]["AC"].extend(ac_values)

        quantization_table = chroma_qtable
        channel_type = "Chroma"

    # the huffman encoding process for AC/DC component
    YCbCr_huffman_table, YCbCr_encoded_data = encode_compressed_data(YCbCr_rawdata)

    # adding all the various markers of jpeg
    # jpeg_datapack = assemble(luma_qtable, chroma_qtable, YCbCr_huffman_table, YCbCr_encoded_data)


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

import cProfile


# cProfile.run('compress("C:/Users/yeapz/OneDrive/Desktop/Python/PyImagery/src/pyimagery/jpeg/test.jpg")', sort="cumtime")
# compress("C:/Users/yeapz/OneDrive/Desktop/Python/PyImagery/src/pyimagery/jpeg/test.jpg")
