from PIL import Image
import numpy as np
import pytest

from jpeg_compressor import *


@pytest.fixture
def img_array_8x8_section():
    return np.array(
        [
            [-76, -73, -67, -62, -58, -67, -64, -55],
            [-65, -69, -73, -38, -19, -43, -59, -56],
            [-66, -69, -60, -15, 16, -24, -62, -55],
            [-65, -70, -57, -6, 26, -22, -58, -59],
            [-61, -67, -60, -24, -2, -40, -60, -58],
            [-49, -63, -68, -58, -51, -60, -70, -53],
            [-43, -57, -64, -69, -73, -67, -63, -45],
            [-41, -49, -59, -60, -63, -52, -50, -34],
        ]
    )


@pytest.fixture
def img_array():
    return np.asarray(Image.open("test.jpg").convert("YCbCr").split()[0])


def test_downsampling(img_array):
    img_array = pad_array(img_array)
    row, col = img_array.shape
    shrunk_row, shrunk_col = row // 2, col // 2
    shrunk_img = downsample(img_array)
    assert shrunk_img.shape == (shrunk_row, shrunk_col)
