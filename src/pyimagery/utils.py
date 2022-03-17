import numpy as np
from numpy.lib.stride_tricks import as_strided as ast


def chunkify(arr: np.ndarray, chunk_size: tuple):
    shape = (arr.shape[0] // chunk_size[0], arr.shape[1] // chunk_size[1]) + chunk_size
    strides = (chunk_size[0] * arr.strides[0], chunk_size[1] * arr.strides[1]) + arr.strides
    return ast(arr, shape=shape, strides=strides)


def get_padded_size(arr_shape: tuple, pad_to: int) -> tuple:
    return (arr_shape[0] + (pad_to - arr_shape[0] % pad_to), arr_shape[1] + (pad_to - arr_shape[1] % pad_to))
