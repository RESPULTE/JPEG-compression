from typing import List, Tuple
from functools import lru_cache


def encode(dataset: List[int]) -> List[Tuple[int, int]]:
    dataset_size = len(dataset)
    encoded_data = []
    curr_index = 0

    while curr_index < dataset_size:
        curr_elem = dataset[curr_index]
        curr_index += 1
        count = 1

        if curr_elem != 0:
            encoded_data.append((0, curr_elem))
            continue

        while curr_index < dataset_size and curr_elem == dataset[curr_index]:
            curr_index += 1
            count += 1

        if curr_index == dataset_size:
            encoded_data.append((0, 0))
            break

        while count > 15:
            encoded_data.append((15, 0))
            count -= 15

        encoded_data.append((count, dataset[curr_index]))
        curr_index += 1

    return encoded_data


def izigzag(dataset: List[int], row: int = 8, col: int = 8) -> List[int]:
    # dataset_size = sum(1 for i in range(len(dataset)) for j in range(len(dataset[i])))
    # if row * col != dataset_size:
    #     raise ValueError(f"cannot form {row}x{col} matrix with datset of size '{dataset_size}'")
    index_list = generate_dzigzag_index(row, col)
    matrix = [[None for _ in range(col)] for _ in range(row)]
    for index, (i, j) in enumerate(index_list):
        matrix[i][j] = dataset[index]

    return matrix


def zigzag(dataset: List[List[int]], row: int = 8, col: int = 8) -> List[int]:
    index_list = generate_dzigzag_index(row, col)
    try:
        return [dataset[i][j] for (i, j) in index_list]
    except IndexError:
        raise ValueError("dataset is not symmetrical, row & column does not match")


@lru_cache
def generate_dzigzag_index(row: int, col: int) -> List[Tuple[int, int]]:
    """generate all the required indexes to turn a 2D array
       into a 1D array in a zig-zaggy fashion in a shape of a sideway Z

    Args:
        row (int): number of row in the 2D array
        col (int): number of columns in the 2D array

    Returns:
        List[Tuple[int, int]]: a list of tupe of indexes, (i, j)
    """
    index_list = [[] for _ in range(row + col - 1)]

    for i in range(row):
        for j in range(col):
            index_sum = i + j
            if index_sum % 2 == 0:
                index_list[index_sum].insert(0, (i, j))
                continue
            index_list[index_sum].append((i, j))

    return [(i, j) for coor in index_list for (i, j) in coor]
