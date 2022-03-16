import numpy as np

from lookup_table import WEIGHTAGE_TABLE, NORMALIZED_COSINE_TABLE


def dct_1D(arr: np.ndarray):
    v0 = arr[0] + arr[7]
    v1 = arr[1] + arr[6]
    v2 = arr[2] + arr[5]
    v3 = arr[3] + arr[4]
    v4 = arr[3] - arr[4]
    v5 = arr[2] - arr[5]
    v6 = arr[1] - arr[6]
    v7 = arr[0] - arr[7]

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


def idct_1D(arr: np.ndarray):
    v15 = arr[0] / NORMALIZED_COSINE_TABLE[0]
    v26 = arr[1] / NORMALIZED_COSINE_TABLE[1]
    v21 = arr[2] / NORMALIZED_COSINE_TABLE[2]
    v28 = arr[3] / NORMALIZED_COSINE_TABLE[3]
    v16 = arr[4] / NORMALIZED_COSINE_TABLE[4]
    v25 = arr[5] / NORMALIZED_COSINE_TABLE[5]
    v22 = arr[6] / NORMALIZED_COSINE_TABLE[6]
    v27 = arr[7] / NORMALIZED_COSINE_TABLE[7]

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
        WEIGHTAGE_TABLE[1] * WEIGHTAGE_TABLE[4]
        - WEIGHTAGE_TABLE[1] * WEIGHTAGE_TABLE[3]
        - WEIGHTAGE_TABLE[3] * WEIGHTAGE_TABLE[4]
    )
    v14 = (v18 - v20 * WEIGHTAGE_TABLE[1]) / (
        WEIGHTAGE_TABLE[1] * WEIGHTAGE_TABLE[4]
        - WEIGHTAGE_TABLE[1] * WEIGHTAGE_TABLE[3]
        - WEIGHTAGE_TABLE[3] * WEIGHTAGE_TABLE[4]
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
